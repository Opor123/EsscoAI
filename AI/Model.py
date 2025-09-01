import json
import pandas as pd
import argparse
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
import logging
import re
import time
from functools import lru_cache
import hashlib

try:
    from .feedback_store import FeedbackStore
except ImportError:
    from feedback_store import FeedbackStore

QA=[
    ("question", "answer"),
    ("Question", "Answer"),
    ("prompt", "completion"),
    ("input", "output"),
    ("instruction", "response"),
    ("user", "assistant"),
    ("q", "a"),
]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory (main folder)"""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent


def resolve_data_path(relative_path: str) -> Path:
    """Resolve data path relative to project root"""
    project_root = get_project_root()
    return project_root / relative_path

FEEDBACK_FILE=Path("./Data/feedback.jsonl")
@dataclass
class RetrievalResult:
    answer: str
    question: str
    similarity_score: float
    index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and clean result data"""
        self.similarity_score = max(0.0, min(1.0, self.similarity_score))
        self.answer = str(self.answer).strip()
        self.question = str(self.question).strip()


class QARetriever:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = self._build_config(config)
        self.vectorizer = None
        self.X = None
        self.df = None
        self.q_col = None
        self.a_col = None
        self._cache_file = None
        self._data_hash = None

    def _build_config(self, user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build configuration with proper merging and validation"""
        default_config = {
            'vectorizer_params': {
                'analyzer': 'word',
                'ngram_range': (1, 3),
                'max_features': 100000,
                'min_df': 2,  # Increased to reduce noise
                'max_df': 0.9,  # More restrictive
                'stop_words': 'english',
                'lowercase': True,
                'token_pattern': r'\b\w+\b',
                'sublinear_tf': True,  # Better for large datasets
                'use_idf': True,
                'smooth_idf': True
            },
            'similarity_threshold': 0.15,  # Slightly higher threshold
            'preprocessing': True,
            'use_cache': True,
            'max_query_length': 500,
            'min_answer_length': 3,
            'fuzzy_matching': True,
            'boost_exact_matches': True
        }

        if user_config:
            # Deep merge for nested dictionaries
            config = default_config.copy()
            for key, value in user_config.items():
                if key == 'vectorizer_params' and isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
            return config
        return default_config

    @lru_cache(maxsize=128)
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing with caching"""
        if not isinstance(text, str):
            text = str(text)

        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)

        # Technical specifications normalization
        normalizations = [
            (r'(\d+)\s*%', r'\1%'),
            (r'(\d+)\s*(hz|khz|mhz|ghz)\b', r'\1\2', re.I),
            (r'(\d+)\s*(v|kv|mv)\b', r'\1\2', re.I),
            (r'(\d+)\s*(va|kva|mva)\b', r'\1\2', re.I),
            (r'(\d+)\s*(w|kw|mw|watt|watts)\b', r'\1\2', re.I),
            (r'(\d+)\s*(amp|amps|ampere|amperes|a)\b', r'\1amp', re.I),
            (r'(\d+)\s*(rpm)\b', r'\1\2', re.I),
            (r'(\d+)\s*(mm|cm|m|km|inch|inches|ft|feet)\b', r'\1\2', re.I),
        ]

        for pattern, replacement, *flags in normalizations:
            flags = flags[0] if flags else 0
            text = re.sub(pattern, replacement, text, flags=flags)

        # Remove excessive punctuation but keep useful ones
        text = re.sub(r'[^\w\s.,%/+-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        return text.lower() if self.config['preprocessing'] else text

    def _compute_data_hash(self, path: Path) -> str:
        """Compute hash of dataset for cache validation"""
        hasher = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_cache_path(self, data_path: Path) -> Path:
        """Generate cache file path"""
        cache_dir = data_path.parent / '.cache'
        cache_dir.mkdir(exist_ok=True)
        cache_name = f"{data_path.stem}_{self._data_hash}.pkl"
        return cache_dir / cache_name

    def _save_cache(self):
        """Save processed data to cache"""
        if not self.config['use_cache'] or self._cache_file is None:
            return

        try:
            cache_data = {
                'df': self.df,
                'q_col': self.q_col,
                'a_col': self.a_col,
                'vectorizer': self.vectorizer,
                'X': self.X,
                'config': self.config
            }

            with open(self._cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Cache saved to {self._cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_cache(self) -> bool:
        """Load processed data from cache"""
        if not self.config['use_cache'] or self._cache_file is None:
            return False

        if not self._cache_file.exists():
            return False

        try:
            with open(self._cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            # Validate cache compatibility
            if cache_data.get('config', {}).get('vectorizer_params') != self.config['vectorizer_params']:
                logger.info("Cache config mismatch, rebuilding...")
                return False

            self.df = cache_data['df']
            self.q_col = cache_data['q_col']
            self.a_col = cache_data['a_col']
            self.vectorizer = cache_data['vectorizer']
            self.X = cache_data['X']

            logger.info(f"Cache loaded from {self._cache_file}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False

    def load_dataset(self, path: Path) -> Tuple[pd.DataFrame, str, str]:
        """Enhanced dataset loading with better error handling"""
        if not path.exists():
            raise FileNotFoundError(f'Dataset not found at {path}')

        logger.info(f'Loading dataset from {path}')
        records = []
        invalid_lines = 0
        line_errors = []

        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    if isinstance(record, dict) and record:  # Ensure non-empty dict
                        records.append(record)
                    else:
                        invalid_lines += 1
                        if line_num <= 5:  # Log first few errors
                            line_errors.append(f'Line {line_num}: Expected non-empty dict, got {type(record)}')
                except json.JSONDecodeError as e:
                    invalid_lines += 1
                    if line_num <= 5:
                        line_errors.append(f'Line {line_num}: JSON decode error - {str(e)[:50]}')
                    continue

        # Log sample errors
        for error in line_errors:
            logger.warning(error)
        if invalid_lines > 5:
            logger.warning(f'... and {invalid_lines - 5} more invalid lines')

        if not records:
            raise ValueError('No valid JSON records found in the dataset')

        logger.info(f'Loaded {len(records)} valid records, skipped {invalid_lines} invalid lines')

        df = pd.DataFrame(records)
        q_col, a_col = self._infer_columns(df)
        df = self._clean_dataframe(df, q_col, a_col)

        logger.info(f'Dataset ready: {len(df)} Q&A pairs')
        return df, q_col, a_col

    def _infer_columns(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Enhanced column inference with better patterns"""
        exact_candidates = [
            ("question", "answer"),
            ("Question", "Answer"),
            ("prompt", "completion"),
            ("input", "output"),
            ("instruction", "response"),
            ("user", "assistant"),
            ("q", "a"),
            ("query", "reply"),
            ("text", "label"),
            ("human", "assistant"),
            ("user_input", "bot_response")
        ]

        # Exact match first
        for q_cand, a_cand in exact_candidates:
            if q_cand in df.columns and a_cand in df.columns:
                logger.info(f'Found exact column match: {q_cand} -> {a_cand}')
                return q_cand, a_cand

        # Case-insensitive matching
        cols_lower = {c.lower(): c for c in df.columns}
        q_patterns = ["question", "prompt", "input", "instruction", "user", "query", "q", "text", "human"]
        a_patterns = ["answer", "completion", "output", "response", "assistant", "reply", "a", "label", "target"]

        for q_pattern in q_patterns:
            for a_pattern in a_patterns:
                if q_pattern in cols_lower and a_pattern in cols_lower:
                    q_col, a_col = cols_lower[q_pattern], cols_lower[a_pattern]
                    logger.info(f'Inferred columns (case-insensitive): {q_col} -> {a_col}')
                    return q_col, a_col

        # Fuzzy matching
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in q_patterns):
                for other_col in df.columns:
                    if col != other_col and any(pattern in other_col.lower() for pattern in a_patterns):
                        logger.info(f"Fuzzy matched columns: {col} -> {other_col}")
                        return col, other_col

        # Last resort: use first two text columns
        text_cols = [c for c in df.columns if df[c].dtype == 'object']
        if len(text_cols) >= 2:
            logger.warning(f"Using first two text columns as fallback: {text_cols[0]} -> {text_cols[1]}")
            return text_cols[0], text_cols[1]

        raise ValueError(
            f"Could not infer question/answer columns from: {list(df.columns)}. "
            f"Supported patterns: {q_patterns} -> {a_patterns}"
        )

    def _clean_dataframe(self, df: pd.DataFrame, q_col: str, a_col: str) -> pd.DataFrame:
        """Enhanced dataframe cleaning with better validation"""
        original_len = len(df)

        # Convert to string and handle NaN
        df[q_col] = df[q_col].astype(str)
        df[a_col] = df[a_col].astype(str)

        # Remove rows with missing or empty content
        df = df[
            (df[q_col] != 'nan') &
            (df[a_col] != 'nan') &
            (df[q_col].str.strip() != '') &
            (df[a_col].str.strip() != '')
            ]

        # Length-based filtering
        if self.config.get('min_answer_length', 0) > 0:
            min_len = self.config['min_answer_length']
            df = df[df[a_col].str.len() >= min_len]

        # Remove duplicates (case-insensitive)
        df['_q_lower'] = df[q_col].str.lower().str.strip()
        df = df.drop_duplicates(subset=['_q_lower'], keep='first')
        df = df.drop(columns=['_q_lower'])

        # Remove very short questions (likely noise)
        df = df[df[q_col].str.len() >= 5]

        df = df.reset_index(drop=True)

        removed = original_len - len(df)
        if removed > 0:
            logger.info(f"Cleaned dataset: removed {removed} invalid/duplicate rows ({removed / original_len:.1%})")

        if len(df) == 0:
            raise ValueError("No valid data remaining after cleaning")

        return df

    def build_index(self, df: pd.DataFrame, q_col: str) -> Tuple[TfidfVectorizer, Any]:
        """Enhanced index building with progress tracking"""
        logger.info('Building TF-IDF index...')
        start_time = time.time()

        questions = df[q_col].astype(str).tolist()

        if self.config['preprocessing']:
            logger.info('Preprocessing questions...')
            questions = [self.preprocess_text(q) for q in questions]

        # Filter out very short processed questions
        valid_questions = []
        valid_indices = []
        for i, q in enumerate(questions):
            if len(q.split()) >= 2:  # At least 2 words
                valid_questions.append(q)
                valid_indices.append(i)

        if len(valid_questions) < len(questions):
            logger.info(f"Filtered out {len(questions) - len(valid_questions)} very short questions")
            df = df.iloc[valid_indices].reset_index(drop=True)
            questions = valid_questions

        vectorizer = TfidfVectorizer(**self.config['vectorizer_params'])


        try:
            X = vectorizer.fit_transform(questions)
            build_time = time.time() - start_time

            logger.info(f"Index built in {build_time:.2f}s: {X.shape[0]} documents, {X.shape[1]} features")
            logger.info(f"Sparsity: {(1.0 - X.nnz / (X.shape[0] * X.shape[1])):.2%}")

            return vectorizer, X, df
        except Exception as e:
            logger.error(f'Failed to build index: {e}')
            raise

    def _boost_exact_matches(self, query: str, similarities: List[float]) -> List[float]:
        """Boost similarity scores for exact phrase matches"""
        if not self.config.get('boost_exact_matches', True):
            return similarities

        query_lower = query.lower()
        boosted_similarities = similarities.copy()

        for i, question in enumerate(self.df[self.q_col]):
            question_lower = str(question).lower()

            # Exact phrase match bonus
            if query_lower in question_lower or question_lower in query_lower:
                boosted_similarities[i] *= 1.3  # 30% boost

            # Word overlap bonus
            query_words = set(query_lower.split())
            question_words = set(question_lower.split())
            overlap = len(query_words & question_words) / len(query_words | question_words)
            if overlap > 0.5:
                boosted_similarities[i] *= (1 + overlap * 0.2)  # Up to 20% boost

        return boosted_similarities

    def retrieve(self, query: str, top_k: int = 3,user_profile:Optional[Dict[str,Any]]=None) -> List[RetrievalResult]:
        """Enhanced retrieval with better ranking and filtering"""
        if self.vectorizer is None or self.X is None:
            raise ValueError("Index not built. Call build_index first.")

        # Input validation
        if not query or not query.strip():
            return []

        query = query.strip()
        if len(query) > self.config.get('max_query_length', 500):
            query = query[:self.config['max_query_length']]
            logger.warning(f"Query truncated to {self.config['max_query_length']} characters")

        processed_query = self.preprocess_text(query) if self.config['preprocessing'] else query

        try:
            # Vectorize query
            q_vec = self.vectorizer.transform([processed_query])

            # Compute similarities
            similarities = cosine_similarity(q_vec, self.X)[0]

            # Apply boosting for exact matches
            if self.config.get('boost_exact_matches', True):
                similarities = self._boost_exact_matches(query, similarities.tolist())

            # Get top results
            top_indices = sorted(range(len(similarities)),
                                 key=lambda i: similarities[i],
                                 reverse=True)[:top_k * 2]  # Get extra for filtering

            results = []
            seen_answers = set()  # Avoid duplicate answers

            for idx in top_indices:
                similarity_score = float(similarities[idx])

                if similarity_score < self.config['similarity_threshold']:
                    continue

                answer = str(self.df.iloc[idx][self.a_col])
                question = str(self.df.iloc[idx][self.q_col])

                # Skip duplicate answers
                answer_key = answer.lower().strip()
                if answer_key in seen_answers:
                    continue
                seen_answers.add(answer_key)

                result = RetrievalResult(
                    answer=answer,
                    question=question,
                    similarity_score=similarity_score,
                    index=idx,
                    metadata={
                        'processed_query': processed_query,
                        'original_query': query
                    }
                )
                results.append(result)

                if len(results) >= top_k:
                    break

                if user_profile:
                    result.similarity_score=self._apply_personalization(result,user_profile)

            return results

        except Exception as e:
            logger.error(f'Retrieval failed: {e}')
            return []

    def get_best_answer(self, query: str) -> Optional[str]:
        """Get the best answer with confidence checking"""
        results = self.retrieve(query, top_k=1)
        if results and results[0].similarity_score >= self.config['similarity_threshold']:
            return results[0].answer
        return None

    def batch_retrieve(self, queries: List[str], top_k: int = 1) -> List[List[RetrievalResult]]:
        """Batch retrieval for multiple queries"""
        return [self.retrieve(query, top_k) for query in queries]

    def add_qa_pair(self, question: str, answer: str) -> bool:
        """Add a new Q&A pair to the system"""
        try:
            new_row = {self.q_col: question, self.a_col: answer}
            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

            # Rebuild index
            self.vectorizer, self.X, self.df = self.build_index(self.df, self.q_col)
            logger.info(f"Added new Q&A pair. Dataset now has {len(self.df)} pairs.")
            return True
        except Exception as e:
            logger.error(f"Failed to add Q&A pair: {e}")
            return False

    def get_similar_questions(self, query: str, top_k: int = 5) -> List[str]:
        """Get similar questions without answers"""
        results = self.retrieve(query, top_k)
        return [result.question for result in results]

    def load_and_build(self, data_path: Path, force_rebuild: bool = False):
        """Load dataset and build index with caching support"""
        # Compute data hash for cache validation
        self._data_hash = self._compute_data_hash(data_path)
        self._cache_file = self._get_cache_path(data_path)

        # Try to load from cache first
        if not force_rebuild and self._load_cache():
            return

        # Load and process data
        self.df, self.q_col, self.a_col = self.load_dataset(data_path)
        self.vectorizer, self.X, self.df = self.build_index(self.df, self.q_col)

        # Save to cache
        self._save_cache()

    def export_results(self, queries: List[str], output_path: Path):
        """Export retrieval results to file"""
        results_data = []
        for query in queries:
            results = self.retrieve(query, top_k=3)
            results_data.append({
                'query': query,
                'results': [
                    {
                        'question': r.question,
                        'answer': r.answer,
                        'similarity': r.similarity_score,
                        'index': r.index
                    } for r in results
                ]
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results exported to {output_path}")

    def _apply_personalization(self,result:RetrievalResult,profile:Dict[str,Any])->float:
        score=result.similarity_score

        if "interests" in profile:
            for interest in profile['interests']:
                if interest.lower() in result.answer.lower() or interest.lower() in result.question.lower():
                    score*=1.2

        if profile.get('role')=='technician':
            if any(word in result.answer.lower() for word in ["spec", "voltage", "amp", "kw"]):
                score*=1.1

        if profile.get('role')=='manager':
            if any(word in result.answer.lower() for word in ['summary','overview','cost']):
                score*=1.1

        return score

    def fit_with_extra_pairs(self,base_df:pd.DataFrame,q_col:str,a_col:str,extra_pairs:List[Dict[str,str]] | None=None):
        df=base_df.copy()
        if extra_pairs:
            extra_df=pd.DataFrame(extra_pairs)
            keep=[c for c in [q_col,a_col] if c in extra_df.columns]
            if keep:
                extra_df=extra_df[keep]
                df=pd.concat([df,extra_df],ignore_index=True)


        df=self._clean_dataframe(df,q_col,a_col)
        self.df=df
        self.q_col,self.a_col=q_col,a_col
        self.vectorizer,self.X, self.df=self.build_index(self.df,self.q_col)
        self._save_cache()

    def retrain_with_feedback(self,feedback_path:Path):

        if self.df is None or self.q_col is None or self.a_col is None:
            raise RuntimeError("Call load_and_build(...) before retraining with feedback.")
        store=FeedbackStore(feedback_path)
        extra=store.load_as_qa_pairs(q_key=self.q_col,a_key=self.a_col)
        self.fit_with_extra_pairs(self.df,self.q_col,self.a_col,extra_pairs=extra)


def create_interactive_session(retriever: QARetriever,top_k:int=3):
    """Enhanced interactive Q&A session with better UX"""
    print(f"\n{'=' * 60}")
    print(f"🤖 Enhanced Q&A Retrieval System")
    print(f"Dataset: {len(retriever.df):,} Q&A pairs")

    print(f"Vocabulary: {len(retriever.vectorizer.vocabulary_):,} terms")
    print(f"Questions: '{retriever.q_col}' | Answers: '{retriever.a_col}'")
    print(f"{'=' * 60}")
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - ':help' for detailed help")
    print("  - ':stats' for dataset statistics")
    print("  - ':config' to view configuration")
    print("  - ':test <query>' to test specific query")
    print("  - ':similar <query>' to find similar questions only")
    print("  - ':add' to add a new Q&A pair")
    print("  - ':retrain' to rebuild the index from collected feedback")
    print("  - 'quit', 'exit', or Ctrl+C to quit")
    print(f"{'=' * 60}\n")

    query_count = 0
    session_start = time.time()

    try:
        while True:
            try:
                query = input('🔍 You: ').strip()

                if not query:
                    continue

                # Handle commands
                if query.lower() in ['quit', 'exit', 'bye']:
                    break
                elif query == ':help':
                    print_help()
                    continue
                elif query == ':stats':
                    print_stats(retriever)
                    continue
                elif query == ':config':
                    print_config(retriever)
                    continue
                elif query.startswith(':test '):
                    test_query = query[6:]
                    print_test_results(retriever, test_query)
                    continue
                elif query.startswith(':similar '):
                    similar_query = query[9:]
                    print_similar_questions(retriever, similar_query)
                    continue
                elif query == ':add':
                    add_qa_pair(retriever)
                    continue
                elif query==':retrain':
                    try:
                        retriever.retrain_with_feedback(FEEDBACK_FILE)
                        print("✅ Retrained on collected feedback.\n")
                    except Exception as e:
                        print(f"❌ Retrain failed: {e}\n")
                    continue

                if query.lower() in ['hi','hello','hey']:
                    print("🤖 Bot: Hi there! Ask me anything about our products or services.")
                    continue

                # Process query
                start_time = time.time()
                results = retriever.retrieve(query, top_k=top_k)
                query_time = time.time() - start_time
                query_count += 1

                if not results:
                    print("🤔 Bot: I couldn't find a good answer for that question.")
                    print("    💡 Try rephrasing, using different keywords, or check ':help'\n")
                    continue



                # Display best answer
                best = results[0]
                confidence_emoji = "🎯" if best.similarity_score > 0.7 else "📍" if best.similarity_score > 0.4 else "🔍"

                print(f"🤖 Bot: {best.answer}")
                print(f"    {confidence_emoji} Confidence: {best.similarity_score:.1%} | "
                      f"Query time: {query_time * 1000:.1f}ms")
                try:
                    fb = input("   Was this helpful? [y/n] (Enter to skip) ").strip().lower()
                    if fb in ('y','yes','n','no'):
                        store=FeedbackStore(FEEDBACK_FILE)
                        if fb in ('y','yes'):
                            store.append({
                                'query':query,
                                'model_answer':best.answer,
                                'label':'up',
                                'correct_answer':None
                            })
                            print("   👍 Thanks! Recorded.\n")
                        else:
                            correction = input("   Provide the correct/ideal answer (optional): ").strip()
                            store.append({
                                'query': query,
                                'model_answer': best.answer,
                                'label': 'down',
                                'correct_answer': correction or None
                            })
                            print("   👌 Got it. Feedback saved.\n")
                except Exception as _e:
                    logger.debug(f'Feedback capture skipped:{_e}')
                # Show related questions if multiple results
                if len(results) > 1:
                    print(f"\n    📋 Related questions:")
                    for i, result in enumerate(results[1:], 1):
                        truncated_q = result.question[:70] + ('...' if len(result.question) > 70 else '')
                        print(f"    {i}. {truncated_q} ({result.similarity_score:.1%})")

                print()

            except (KeyboardInterrupt, EOFError):
                break
            except Exception as e:
                logger.error(f"Error during interaction: {e}")
                print("❌ An error occurred. Please try again.\n")

    except Exception as e:
        logger.error(f"Session error: {e}")
    finally:
        session_time = time.time() - session_start
        print(f"\n👋 Session complete!")
        print(f"   Queries processed: {query_count}")
        print(f"   Session time: {session_time:.1f}s")
        print(f"   Thanks for using the Enhanced Q&A System!")


def print_help():
    """Enhanced help information"""
    print("""
📖 Enhanced Help Guide:

🔍 Search Tips:
  - Use specific keywords related to your domain
  - Try different phrasings if results aren't satisfactory
  - Include technical terms or model numbers for better matching
  - Shorter, focused questions often work better than long ones

📊 Understanding Results:
  - Confidence scores: >70% = excellent, >40% = good, <40% = uncertain
  - Related questions help you explore similar topics
  - Query time indicates system performance

🛠️ Commands:
  - ':test <query>' - Test a query and see detailed matching info
  - ':similar <query>' - Find similar questions without answers
  - ':add' - Add your own Q&A pair to improve the system
  - ':stats' - View dataset statistics and performance metrics
  - ':config' - View current system configuration

💡 Pro Tips:
  - If confidence is low, try rephrasing with synonyms
  - Use domain-specific terminology when available
  - Check related questions for alternative phrasings
""")


def print_stats(retriever: QARetriever):
    """Enhanced statistics display"""
    df = retriever.df
    q_lens = df[retriever.q_col].str.len()
    a_lens = df[retriever.a_col].str.len()

    print(f"\n📊 Enhanced Dataset Statistics:")
    print(f"  📋 Total Q&A pairs: {len(df):,}")
    print(f"  📝 Columns: '{retriever.q_col}' -> '{retriever.a_col}'")
    print(f"  📏 Question lengths: avg={q_lens.mean():.1f}, min={q_lens.min()}, max={q_lens.max()}")
    print(f"  📄 Answer lengths: avg={a_lens.mean():.1f}, min={a_lens.min()}, max={a_lens.max()}")

    if retriever.vectorizer:
        vocab_size = len(retriever.vectorizer.vocabulary_)
        print(f"  🔤 Vocabulary size: {vocab_size:,}")
        print(f"  🎯 Feature matrix: {retriever.X.shape[0]:,} × {retriever.X.shape[1]:,}")
        print(f"  💾 Matrix sparsity: {(1.0 - retriever.X.nnz / (retriever.X.shape[0] * retriever.X.shape[1])):.2%}")

    print()


def print_config(retriever: QARetriever):
    """Enhanced configuration display"""
    print(f"\n⚙️  System Configuration:")
    for key, value in retriever.config.items():
        if isinstance(value, dict):
            print(f"  📁 {key}:")
            for sub_key, sub_value in value.items():
                print(f"    • {sub_key}: {sub_value}")
        else:
            print(f"  • {key}: {value}")
    print()


def print_test_results(retriever: QARetriever, query: str):
    """Test a query and show detailed results"""
    print(f"\n🧪 Testing query: '{query}'")
    results = retriever.retrieve(query, top_k=5)

    if not results:
        print("❌ No results found above threshold")
        return

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Similarity: {result.similarity_score:.3f}")
        print(f"   Q: {result.question}")
        print(f"   A: {result.answer[:100]}{'...' if len(result.answer) > 100 else ''}")


def print_similar_questions(retriever: QARetriever, query: str):
    """Show similar questions without answers"""
    print(f"\n🔍 Questions similar to: '{query}'")
    questions = retriever.get_similar_questions(query, top_k=5)

    if not questions:
        print("❌ No similar questions found")
        return

    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")
    print()


def add_qa_pair(retriever: QARetriever):
    """Interactive Q&A pair addition"""
    print("\n➕ Add New Q&A Pair")
    try:
        question = input("Question: ").strip()
        if not question:
            print("❌ Question cannot be empty")
            return

        answer = input("Answer: ").strip()
        if not answer:
            print("❌ Answer cannot be empty")
            return

        if retriever.add_qa_pair(question, answer):
            print("✅ Q&A pair added successfully!")
        else:
            print("❌ Failed to add Q&A pair")

    except (KeyboardInterrupt, EOFError):
        print("\n❌ Cancelled")


def benchmark_retrieval(retriever: QARetriever, test_queries: List[str]) -> Dict[str, float]:
    """Benchmark retrieval performance"""
    print(f"\n🏃 Benchmarking with {len(test_queries)} queries...")

    start_time = time.time()
    total_results = 0
    successful_queries = 0

    for query in test_queries:
        results = retriever.retrieve(query, top_k=1)
        total_results += len(results)
        if results:
            successful_queries += 1

    total_time = time.time() - start_time

    metrics = {
        'total_time': total_time,
        'avg_time_per_query': total_time / len(test_queries),
        'success_rate': successful_queries / len(test_queries),
        'avg_results_per_query': total_results / len(test_queries)
    }

    print(f"📈 Benchmark Results:")
    print(f"  Total time: {metrics['total_time']:.2f}s")
    print(f"  Avg time per query: {metrics['avg_time_per_query'] * 1000:.1f}ms")
    print(f"  Success rate: {metrics['success_rate']:.1%}")
    print(f"  Avg results per query: {metrics['avg_results_per_query']:.1f}")

    return metrics


def validate_dataset(data_path: Path) -> Dict[str, Any]:
    """Validate dataset quality and structure"""
    print(f"\n🔍 Validating dataset: {data_path}")

    validation_results = {
        'file_exists': data_path.exists(),
        'file_size_mb': data_path.stat().st_size / (1024 * 1024) if data_path.exists() else 0,
        'total_lines': 0,
        'valid_json_lines': 0,
        'empty_lines': 0,
        'json_errors': [],
        'column_candidates': [],
        'sample_records': []
    }

    if not validation_results['file_exists']:
        print(f"❌ File not found: {data_path}")
        return validation_results

    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            validation_results['total_lines'] += 1

            if not line.strip():
                validation_results['empty_lines'] += 1
                continue

            try:
                record = json.loads(line)
                if isinstance(record, dict) and record:
                    validation_results['valid_json_lines'] += 1
                    if len(validation_results['sample_records']) < 3:
                        validation_results['sample_records'].append(record)

                    # Collect column names
                    for key in record.keys():
                        if key not in validation_results['column_candidates']:
                            validation_results['column_candidates'].append(key)

            except json.JSONDecodeError as e:
                if len(validation_results['json_errors']) < 5:
                    validation_results['json_errors'].append(f"Line {line_num}: {str(e)[:50]}")

    # Print validation summary
    print(f"📊 Validation Results:")
    print(f"  File size: {validation_results['file_size_mb']:.1f} MB")
    print(f"  Total lines: {validation_results['total_lines']:,}")
    print(f"  Valid JSON lines: {validation_results['valid_json_lines']:,}")
    print(f"  Empty lines: {validation_results['empty_lines']:,}")
    print(f"  Success rate: {validation_results['valid_json_lines'] / validation_results['total_lines']:.1%}")

    if validation_results['json_errors']:
        print(f"  JSON errors (first 5):")
        for error in validation_results['json_errors']:
            print(f"    - {error}")

    if validation_results['column_candidates']:
        print(f"  Available columns: {validation_results['column_candidates']}")

    if validation_results['sample_records']:
        print(f"  Sample record keys: {list(validation_results['sample_records'][0].keys())}")

    return validation_results


def main():
    """Enhanced main function with comprehensive argument handling"""
    parser = argparse.ArgumentParser(
        description="Enhanced Q&A Retrieval System with Caching and Performance Improvements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python enhanced_qa.py --data ../Data/training_ready.jsonl

  # Performance tuning
  python enhanced_qa.py --data Data/qa.jsonl --threshold 0.3 --max-features 50000

  # Disable preprocessing and caching
  python enhanced_qa.py --data Data/qa.jsonl --no-preprocessing --no-cache

  # Validation mode
  python enhanced_qa.py --validate --data Data/qa.jsonl

  # Benchmark mode
  python enhanced_qa.py --benchmark --data Data/qa.jsonl
        """
    )

    # File and data arguments
    default_path = resolve_data_path('Data/training_ready.jsonl')
    parser.add_argument(
        "--data", type=Path, default=default_path,
        help="Path to JSONL dataset (default: Data/training_ready.jsonl)"
    )

    # Retrieval parameters
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Number of top results to show (default: 3)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.15,
        help="Minimum similarity threshold (default: 0.15)"
    )

    # Vectorizer parameters
    parser.add_argument(
        "--max-features", type=int, default=100000,
        help="Maximum TF-IDF features (default: 100000)"
    )
    parser.add_argument(
        "--ngram-max", type=int, default=3,
        help="Maximum n-gram size (default: 3)"
    )
    parser.add_argument(
        "--min-df", type=int, default=2,
        help="Minimum document frequency (default: 2)"
    )

    # Processing options
    parser.add_argument(
        "--no-preprocessing", action="store_true",
        help="Disable text preprocessing"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable caching system"
    )
    parser.add_argument(
        "--force-rebuild", action="store_true",
        help="Force rebuild index (ignore cache)"
    )

    # Mode options
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate dataset and exit"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run benchmark tests"
    )
    parser.add_argument(
        "--export", type=Path,
        help="Export test results to JSON file"
    )
    parser.add_argument(
        "--feedback", type=Path, default=Path("Data/feedback.jsonl"),
        help="Path to feedback JSONL file (default: data/feedback.jsonl)"
    )
    parser.add_argument(
        "--retrain-now", action="store_true",
        help="Retrain immediately on collected feedback then start"
    )
    # Logging and output
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress non-error output"
    )

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validation mode
    if args.validate:
        validation_results = validate_dataset(args.data)
        return 0 if validation_results['valid_json_lines'] > 0 else 1

    # Build configuration from arguments
    config = {
        'similarity_threshold': args.threshold,
        'preprocessing': not args.no_preprocessing,
        'use_cache': not args.no_cache,
        'vectorizer_params': {
            'analyzer': 'word',
            'ngram_range': (1, args.ngram_max),
            'max_features': args.max_features,
            'min_df': args.min_df,
            'max_df': 0.9,
            'stop_words': 'english',
            'lowercase': True,
            'token_pattern': r'\b\w+\b',
            'sublinear_tf': True,
            'use_idf': True,
            'smooth_idf': True
        }
    }

    try:
        # Initialize retriever
        retriever = QARetriever(config=config)

        # Load and build index
        logger.info("Initializing Q&A retrievl system...")
        retriever.load_and_build(args.data, force_rebuild=args.force_rebuild)

        global FEEDBACK_FILE
        FEEDBACK_FILE=args.feedback
        if args.retrain_now:
            try:
                retriever.retrain_with_feedback(FEEDBACK_FILE)
                logger.info("Retrained on feedback")
            except Exception as e:
                logger.error(f'Retrain failed: {e}')

        # Benchmark mode
        if args.benchmark:
            # Create sample queries from dataset
            sample_questions = retriever.df[retriever.q_col].sample(min(20, len(retriever.df))).tolist()
            benchmark_retrieval(retriever, sample_questions)

            if args.export:
                retriever.export_results(sample_questions, args.export)
                logger.info(f"Benchmark results exported to {args.export}")
            return 0

        # Export mode
        if args.export:
            test_queries = input("Enter test queries (comma-separated): ").split(',')
            test_queries = [q.strip() for q in test_queries if q.strip()]
            retriever.export_results(test_queries, args.export)
            return 0

        # Start interactive session
        create_interactive_session(retriever)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"❌ Error: {e}")
        print(f"💡 Tip: Check if the file path is correct and the file exists")
        return 1
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        print(f"❌ Error: {e}")
        print(f"💡 Tip: Use --validate to check your dataset format")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"❌ Unexpected error: {e}")
        print(f"💡 Tip: Use --verbose for detailed error information")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())