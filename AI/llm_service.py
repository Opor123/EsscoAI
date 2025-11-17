"""
LLM Service for Enhanced Q&A Responses
Supports both OpenAI and Anthropic APIs
"""
import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

# Try importing both providers
try:
    from anthropic import Anthropic, APIError as AnthropicAPIError, RateLimitError as AnthropicRateLimitError

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from openai import OpenAI, APIError as OpenAIAPIError, RateLimitError as OpenAIRateLimitError

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM service"""
    provider: str = "openai"  # "openai" or "anthropic"
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"  # Default OpenAI model
    max_tokens: int = 1000
    confidence_threshold: float = 0.6
    temperature: float = 0.7
    timeout: int = 30
    max_context_results: int = 3
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables"""
        # Determine provider based on available API keys
        provider = os.getenv("ESSCOAI_LLM_PROVIDER", "auto")

        openai_key = os.getenv("ESSCO_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        # Auto-detect provider
        if provider == "auto":
            if openai_key and OPENAI_AVAILABLE:
                provider = "openai"
                api_key = openai_key
            elif anthropic_key and ANTHROPIC_AVAILABLE:
                provider = "anthropic"
                api_key = anthropic_key
            else:
                provider = "openai"  # Default
                api_key = openai_key
        elif provider == "openai":
            api_key = openai_key
        else:
            api_key = anthropic_key

        # Set default model based on provider
        default_model = "gpt-4o-mini" if provider == "openai" else "claude-sonnet-4-20250514"
        model = os.getenv("ESSCOAI_LLM_MODEL", default_model)

        return cls(
            provider=provider,
            api_key=api_key,
            model=model,
            max_tokens=int(os.getenv("ESSCOAI_LLM_MAX_TOKENS", "1000")),
            confidence_threshold=float(os.getenv("ESSCOAI_LLM_CONFIDENCE_THRESHOLD", "0.6")),
            temperature=float(os.getenv("ESSCOAI_LLM_TEMPERATURE", "0.7")),
            timeout=int(os.getenv("ESSCOAI_LLM_TIMEOUT", "30")),
            max_context_results=int(os.getenv("ESSCOAI_LLM_CONTEXT_RESULTS", "3")),
            enabled=os.getenv("ESSCOAI_USE_LLM", "1") == "1"
        )


class LLMService:
    """
    Service for integrating LLM with retrieval system
    Supports both OpenAI and Anthropic APIs
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize LLM client based on provider"""
        if self.config.provider == "openai":
            self._initialize_openai()
        elif self.config.provider == "anthropic":
            self._initialize_anthropic()
        else:
            logger.error(f"Unknown provider: {self.config.provider}")
            self.config.enabled = False

    def _initialize_openai(self):
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI package not available. Install with: pip install openai")
            self.config.enabled = False
            return

        if not self.config.api_key:
            logger.warning("No OpenAI API key found (ESSCO_AI_API_KEY or OPENAI_API_KEY). LLM features disabled.")
            self.config.enabled = False
            return

        try:
            self.client = OpenAI(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            logger.info(f"LLM Service initialized with OpenAI model: {self.config.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.config.enabled = False

    def _initialize_anthropic(self):
        """Initialize Anthropic client"""
        if not ANTHROPIC_AVAILABLE:
            logger.warning("Anthropic package not available. Install with: pip install anthropic")
            self.config.enabled = False
            return

        if not self.config.api_key:
            logger.warning("No ANTHROPIC_API_KEY found. LLM features disabled.")
            self.config.enabled = False
            return

        try:
            self.client = Anthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            logger.info(f"LLM Service initialized with Anthropic model: {self.config.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            self.config.enabled = False

    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.config.enabled and self.client is not None

    def should_use_llm(self, confidence_score: float) -> bool:
        """Determine if LLM should be used based on confidence score"""
        return (
                self.is_available() and
                confidence_score < self.config.confidence_threshold
        )

    def build_context_from_results(self, results: List[Any]) -> str:
        """
        Build context string from retrieval results

        Args:
            results: List of RetrievalResult objects

        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for i, result in enumerate(results[:self.config.max_context_results], 1):
            context_parts.append(
                f"Entry {i} (Relevance: {result.similarity_score:.1%}):\n"
                f"Question: {result.question}\n"
                f"Answer: {result.answer}\n"
            )

        return "\n".join(context_parts)

    def create_prompt(self, user_query: str, context: str) -> str:
        """
        Create the LLM prompt with context

        Args:
            user_query: User's original question
            context: Retrieved context from knowledge base

        Returns:
            Formatted prompt string
        """
        return f"""You are a helpful AI assistant with access to a knowledge base. Your task is to answer the user's question using ONLY the information provided in the knowledge base entries below.

KNOWLEDGE BASE ENTRIES:
{context}

USER'S QUESTION: {user_query}

INSTRUCTIONS:
1. If the knowledge base entries contain relevant information, synthesize a clear, helpful answer
2. If the entries are only partially relevant, use what's available and acknowledge what's missing
3. If the entries don't answer the question, politely say you don't have that specific information
4. Be concise, natural, and accurate
5. Do NOT make up information that isn't in the knowledge base entries
6. If technical details are present (voltages, specifications, model numbers), include them

RESPONSE:"""

    def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided knowledge base entries. Stay grounded in the provided information."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )

            answer = response.choices[0].message.content.strip()

            return {
                'success': True,
                'answer': answer,
                'tokens_used': {
                    'input': response.usage.prompt_tokens,
                    'output': response.usage.completion_tokens,
                    'total': response.usage.total_tokens
                }
            }
        except OpenAIRateLimitError as e:
            logger.error(f"OpenAI rate limit: {e}")
            return {'success': False, 'error': 'rate_limit', 'answer': None}
        except OpenAIAPIError as e:
            logger.error(f"OpenAI API error: {e}")
            return {'success': False, 'error': 'api_error', 'answer': None}
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            return {'success': False, 'error': str(e), 'answer': None}

    def _call_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Call Anthropic API"""
        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            answer = response.content[0].text.strip()

            return {
                'success': True,
                'answer': answer,
                'tokens_used': {
                    'input': response.usage.input_tokens,
                    'output': response.usage.output_tokens,
                    'total': response.usage.input_tokens + response.usage.output_tokens
                }
            }
        except AnthropicRateLimitError as e:
            logger.error(f"Anthropic rate limit: {e}")
            return {'success': False, 'error': 'rate_limit', 'answer': None}
        except AnthropicAPIError as e:
            logger.error(f"Anthropic API error: {e}")
            return {'success': False, 'error': 'api_error', 'answer': None}
        except Exception as e:
            logger.error(f"Anthropic call failed: {e}")
            return {'success': False, 'error': str(e), 'answer': None}

    def enhance_with_llm(
            self,
            user_query: str,
            retrieval_results: List[Any],
            system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhance response using LLM

        Args:
            user_query: User's question
            retrieval_results: List of RetrievalResult objects from retrieval
            system_prompt: Optional custom system prompt

        Returns:
            Dictionary with enhanced response and metadata
        """
        if not self.is_available():
            return {
                'success': False,
                'error': 'LLM service not available',
                'answer': None
            }

        try:
            start_time = time.time()

            # Build context from retrieval results
            context = self.build_context_from_results(retrieval_results)

            # Create prompt
            prompt = self.create_prompt(user_query, context)

            # Call appropriate API
            if self.config.provider == "openai":
                api_result = self._call_openai(prompt)
            else:
                api_result = self._call_anthropic(prompt)

            if not api_result['success']:
                return api_result

            elapsed_time = time.time() - start_time

            logger.info(
                f"LLM enhancement completed in {elapsed_time:.2f}s "
                f"(provider: {self.config.provider}, "
                f"tokens: {api_result['tokens_used']['input']} in, "
                f"{api_result['tokens_used']['output']} out)"
            )

            return {
                'success': True,
                'answer': api_result['answer'],
                'model': self.config.model,
                'provider': self.config.provider,
                'elapsed_time': elapsed_time,
                'tokens_used': api_result['tokens_used'],
                'context_entries': len(retrieval_results[:self.config.max_context_results])
            }

        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'answer': None
            }

    def process_query(
            self,
            user_query: str,
            retrieval_results: List[Any],
            confidence_score: float,
            fallback_answer: str
    ) -> Dict[str, Any]:
        """
        Main processing method - decides whether to use LLM and returns response

        Args:
            user_query: User's question
            retrieval_results: Results from retrieval system
            confidence_score: Confidence score from retrieval
            fallback_answer: Answer to use if LLM fails or isn't needed

        Returns:
            Dictionary with answer and metadata
        """
        # Check if we should use LLM
        if not self.should_use_llm(confidence_score):
            return {
                'answer': fallback_answer,
                'mode': 'retrieval_only',
                'llm_used': False,
                'confidence': confidence_score,
                'reason': 'High confidence - retrieval sufficient'
            }

        # Try LLM enhancement
        llm_result = self.enhance_with_llm(user_query, retrieval_results)

        if llm_result['success']:
            return {
                'answer': llm_result['answer'],
                'mode': 'llm_enhanced',
                'llm_used': True,
                'confidence': confidence_score,
                'llm_metadata': {
                    'provider': llm_result['provider'],
                    'model': llm_result['model'],
                    'elapsed_time': llm_result['elapsed_time'],
                    'tokens': llm_result['tokens_used'],
                    'context_entries': llm_result['context_entries']
                },
                'reason': f'Low confidence - enhanced with {llm_result["provider"].title()}'
            }
        else:
            # LLM failed, fall back to retrieval answer
            logger.warning(f"LLM enhancement failed, using fallback: {llm_result.get('error')}")
            return {
                'answer': fallback_answer,
                'mode': 'retrieval_fallback',
                'llm_used': False,
                'confidence': confidence_score,
                'llm_error': llm_result.get('error'),
                'reason': 'LLM failed - using retrieval fallback'
            }


# Singleton instance
_llm_service = None


def get_llm_service() -> LLMService:
    """Get or create LLM service singleton"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def reset_llm_service():
    """Reset LLM service (useful for testing)"""
    global _llm_service
    _llm_service = None