"""LLM providers and the typed client used by the agent."""

from __future__ import annotations

from researchmind.config import Settings, get_settings
from researchmind.llm.base import LLMError, LLMProvider, LLMResponse, Message
from researchmind.llm.client import BudgetExceeded, LLMClient, UsageCounter
from researchmind.llm.structured import StructuredOutputError

__all__ = [
    "BudgetExceeded",
    "LLMClient",
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "StructuredOutputError",
    "UsageCounter",
    "build_provider",
]


def build_provider(settings: Settings | None = None) -> LLMProvider:
    s = settings or get_settings()
    if s.llm_provider == "ollama":
        from researchmind.llm.ollama import OllamaLLM

        return OllamaLLM(
            s.llm_model,
            s.ollama_url,
            num_ctx=s.ollama_num_ctx,
            think=s.ollama_think,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_output_tokens,
            timeout_s=s.llm_timeout_s,
        )
    if s.llm_provider == "gemini":
        from researchmind.llm.gemini import GeminiLLM

        key = s.gemini_api_key.get_secret_value() if s.gemini_api_key else ""
        return GeminiLLM(
            s.llm_model, key, temperature=s.llm_temperature, max_tokens=s.llm_max_output_tokens
        )
    if s.llm_provider == "openai_compat":
        from researchmind.llm.openai_compat import OpenAICompatLLM

        if not s.openai_compat_base_url:
            raise LLMError("OPENAI_COMPAT_BASE_URL is not set")
        key = s.openai_compat_api_key.get_secret_value() if s.openai_compat_api_key else None
        return OpenAICompatLLM(
            s.llm_model,
            s.openai_compat_base_url,
            key,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_output_tokens,
            timeout_s=s.llm_timeout_s,
        )
    raise LLMError(f"LLM provider {s.llm_provider!r} cannot be built from settings")
