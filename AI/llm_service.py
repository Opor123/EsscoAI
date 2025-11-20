"""
LLM Service for Enhanced Q&A Responses
Supports OpenAI, Anthropic, and LOCAL LLMs (Ollama, LM Studio, etc.)
"""
import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time
import requests

# Try importing cloud providers
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
    provider: str = "ollama"  # "openai", "anthropic", "ollama", "lmstudio", "openai-compatible"
    api_key: Optional[str] = None
    model: str = "llama3.2"  # Default local model
    base_url: str = "http://localhost:11434"  # Ollama default
    max_tokens: int = 1000
    confidence_threshold: float = 0.6
    temperature: float = 0.7
    timeout: int = 60  # Increased for local models
    max_context_results: int = 3
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load configuration from environment variables"""
        provider = os.getenv("ESSCOAI_LLM_PROVIDER", "ollama")

        # Base URLs for different providers
        base_url_map = {
            "ollama": "http://localhost:11434",
            "lmstudio": "http://localhost:1234",
            "openai-compatible": os.getenv("ESSCOAI_LLM_BASE_URL", "http://localhost:8000")
        }

        # Model defaults
        model_defaults = {
            "ollama": "llama3.2",
            "lmstudio": "local-model",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514",
            "openai-compatible": "local-model"
        }

        # Get API key if using cloud providers
        api_key = None
        if provider == "openai":
            api_key = os.getenv("ESSCO_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")

        base_url = os.getenv("ESSCOAI_LLM_BASE_URL", base_url_map.get(provider, "http://localhost:11434"))
        model = os.getenv("ESSCOAI_LLM_MODEL", model_defaults.get(provider, "llama3.2"))

        return cls(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_tokens=int(os.getenv("ESSCOAI_LLM_MAX_TOKENS", "1000")),
            confidence_threshold=float(os.getenv("ESSCOAI_LLM_CONFIDENCE_THRESHOLD", "0.6")),
            temperature=float(os.getenv("ESSCOAI_LLM_TEMPERATURE", "0.7")),
            timeout=int(os.getenv("ESSCOAI_LLM_TIMEOUT", "60")),
            max_context_results=int(os.getenv("ESSCOAI_LLM_CONTEXT_RESULTS", "3")),
            enabled=os.getenv("ESSCOAI_USE_LLM", "1") == "1"
        )


class LLMService:
    """
    Service for integrating LLM with retrieval system
    Supports cloud APIs and local LLMs
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize LLM client based on provider"""
        if self.config.provider in ["ollama", "lmstudio", "openai-compatible"]:
            self._initialize_local()
        elif self.config.provider == "openai":
            self._initialize_openai()
        elif self.config.provider == "anthropic":
            self._initialize_anthropic()
        else:
            logger.error(f"Unknown provider: {self.config.provider}")
            self.config.enabled = False

    def _initialize_local(self):
        """Initialize local LLM (Ollama, LM Studio, etc.)"""
        try:
            # Test connection
            if self.config.provider == "ollama":
                test_url = f"{self.config.base_url}/api/tags"
            else:
                test_url = f"{self.config.base_url}/v1/models"

            response = requests.get(test_url, timeout=5)

            if response.status_code == 200:
                logger.info(f"Local LLM service connected: {self.config.provider} at {self.config.base_url}")
                logger.info(f"Using model: {self.config.model}")
                self.client = "local"  # Marker that we're using local
            else:
                logger.warning(f"Local LLM server responded with status {response.status_code}")
                self.config.enabled = False

        except requests.exceptions.ConnectionError:
            logger.warning(
                f"Cannot connect to {self.config.provider} at {self.config.base_url}. "
                f"Make sure the server is running. LLM features disabled."
            )
            self.config.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize local LLM: {e}")
            self.config.enabled = False

    def _initialize_openai(self):
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI package not available. Install with: pip install openai")
            self.config.enabled = False
            return

        if not self.config.api_key:
            logger.warning("No OpenAI API key found. LLM features disabled.")
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
        """Build context string from retrieval results"""
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
        """Create the LLM prompt with context"""
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

    def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Call Ollama API"""
        try:
            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    }
                },
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                data = response.json()
                answer = data.get("response", "").strip()

                return {
                    'success': True,
                    'answer': answer,
                    'tokens_used': {
                        'input': data.get('prompt_eval_count', 0),
                        'output': data.get('eval_count', 0),
                        'total': data.get('prompt_eval_count', 0) + data.get('eval_count', 0)
                    }
                }
            else:
                logger.error(f"Ollama error: {response.status_code} - {response.text}")
                return {'success': False, 'error': f'http_{response.status_code}', 'answer': None}

        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return {'success': False, 'error': 'timeout', 'answer': None}
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return {'success': False, 'error': str(e), 'answer': None}

    def _call_openai_compatible(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI-compatible API (LM Studio, vLLM, etc.)"""
        try:
            response = requests.post(
                f"{self.config.base_url}/v1/chat/completions",
                json={
                    "model": self.config.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers questions based on provided knowledge base entries."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature
                },
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                data = response.json()
                answer = data['choices'][0]['message']['content'].strip()
                usage = data.get('usage', {})

                return {
                    'success': True,
                    'answer': answer,
                    'tokens_used': {
                        'input': usage.get('prompt_tokens', 0),
                        'output': usage.get('completion_tokens', 0),
                        'total': usage.get('total_tokens', 0)
                    }
                }
            else:
                logger.error(f"OpenAI-compatible API error: {response.status_code}")
                return {'success': False, 'error': f'http_{response.status_code}', 'answer': None}

        except Exception as e:
            logger.error(f"OpenAI-compatible call failed: {e}")
            return {'success': False, 'error': str(e), 'answer': None}

    def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided knowledge base entries."
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
        except Exception as e:
            logger.error(f"Anthropic call failed: {e}")
            return {'success': False, 'error': str(e), 'answer': None}

    def enhance_with_llm(
            self,
            user_query: str,
            retrieval_results: List[Any],
            system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Enhance response using LLM"""
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
            prompt = self.create_prompt(user_query, context)

            # Call appropriate API
            if self.config.provider == "ollama":
                api_result = self._call_ollama(prompt)
            elif self.config.provider in ["lmstudio", "openai-compatible"]:
                api_result = self._call_openai_compatible(prompt)
            elif self.config.provider == "openai":
                api_result = self._call_openai(prompt)
            else:  # anthropic
                api_result = self._call_anthropic(prompt)

            if not api_result['success']:
                return api_result

            elapsed_time = time.time() - start_time

            logger.info(
                f"LLM enhancement completed in {elapsed_time:.2f}s "
                f"(provider: {self.config.provider}, model: {self.config.model}, "
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
        """Main processing method"""
        if not self.should_use_llm(confidence_score):
            return {
                'answer': fallback_answer,
                'mode': 'retrieval_only',
                'llm_used': False,
                'confidence': confidence_score,
                'reason': 'High confidence - retrieval sufficient'
            }

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