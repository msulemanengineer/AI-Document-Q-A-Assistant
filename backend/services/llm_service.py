"""
STEP 7 of the RAG pipeline: send (system prompt + context + question) to an LLM.

Design note: every provider implements the same tiny `LLMProvider` interface
(`generate(system_prompt, user_prompt) -> str`). The rest of the application
only ever talks to that interface, so swapping OpenAI for Anthropic -- or for
a local model -- means adding one small class here and changing one .env line.

`MockProvider` needs no API key. It exists so the PDF -> chunk -> embed ->
FAISS -> retrieve half of the pipeline can be developed and tested offline.
It does NOT generate language; it just returns the retrieved context.
"""

from abc import ABC, abstractmethod
from typing import Optional

from backend.config import settings
from backend.utils.errors import LLMError


class LLMProvider(ABC):
    """The interface every provider must implement."""

    name: str = "base"

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's answer as plain text."""
        raise NotImplementedError


class MockProvider(LLMProvider):
    """Offline fallback. Returns the retrieved context instead of a real answer.

    This keeps the app fully runnable without an API key, and makes it obvious
    in the UI that no real LLM was involved.
    """

    name = "mock"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "[MOCK MODE - no LLM was called]\n\n"
            "Retrieval ran normally and the most relevant passages from your "
            "document are shown under 'Sources' below. To get a real written "
            "answer, set LLM_PROVIDER and LLM_API_KEY in your .env file."
        )


class OpenAIProvider(LLMProvider):
    """OpenAI, and anything that speaks the OpenAI chat-completions API.

    Setting LLM_BASE_URL points the same client at Groq, Together, OpenRouter,
    a local Ollama server, etc. -- no code change needed.
    """

    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str = ""):
        if not api_key:
            raise LLMError(
                "LLM_API_KEY is not set. Add it to your .env file, "
                "or set LLM_PROVIDER=mock to run without an LLM."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("The 'openai' package is not installed.") from exc

        self.model = model or "gpt-4o-mini"
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                # temperature 0 makes the model stick closely to the context
                # and keeps answers reproducible -- both good for RAG.
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            raise LLMError(f"The OpenAI request failed: {exc}") from exc


class AnthropicProvider(LLMProvider):
    """Anthropic Claude via the Messages API."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMError(
                "LLM_API_KEY is not set. Add it to your .env file, "
                "or set LLM_PROVIDER=mock to run without an LLM."
            )
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("The 'anthropic' package is not installed.") from exc

        self.model = model or "claude-haiku-4-5-20251001"
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            # Claude takes the system prompt as its own parameter, not as a
            # message -- one of the few real differences between the two APIs.
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            parts = [block.text for block in response.content if block.type == "text"]
            return "".join(parts).strip()
        except Exception as exc:
            raise LLMError(f"The Anthropic request failed: {exc}") from exc


# Cache the provider so we do not rebuild an HTTP client on every request.
_provider: Optional[LLMProvider] = None


def get_llm_provider(force_reload: bool = False) -> LLMProvider:
    """Build the provider named by LLM_PROVIDER in .env.

    Raises LLMError with an actionable message if the provider is misconfigured
    (unknown name, missing API key, missing package). The caller turns that into
    an HTTP 502, so the user is told exactly what to fix instead of getting a
    silently wrong answer. To run with no API key at all, set LLM_PROVIDER=mock.
    """
    global _provider
    if _provider is not None and not force_reload:
        return _provider

    provider_name = (settings.LLM_PROVIDER or "mock").lower()

    if provider_name == "mock":
        _provider = MockProvider()
    elif provider_name == "openai":
        _provider = OpenAIProvider(
            settings.LLM_API_KEY, settings.LLM_MODEL, settings.LLM_BASE_URL
        )
    elif provider_name == "anthropic":
        _provider = AnthropicProvider(settings.LLM_API_KEY, settings.LLM_MODEL)
    else:
        raise LLMError(
            f"Unknown LLM_PROVIDER '{provider_name}'. "
            f"Use one of: mock, openai, anthropic."
        )

    return _provider
