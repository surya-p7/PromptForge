MAX_RETRIES = 2           # retry attempts
TIMEOUT_MS = 15000         # 6 second timeout


# app/llm/provider_router.py
from typing import List
from .base import BaseLLMProvider
from .provider_loader import build_provider_chain


class ProviderRouter:
    """
    Routes generation requests across a chain of providers with fallback.

    Example chain (auto-detected):
      [GeminiProvider, OllamaProvider, MockProvider]
    """

    def __init__(self, providers: List[BaseLLMProvider] | None = None):
        if providers is None:
            providers, report = build_provider_chain()
            self.providers = providers
            self.report = report
        else:
            self.providers = providers
            self.report = {}

    def safe_generate(self, prompt: str, **params) -> dict:
        last_error: str | None = None
        for provider in self.providers:
            name = provider.__class__.__name__

            for attempt in range(1, MAX_RETRIES + 2):  # initial try + retries
                try:
                    resp = provider.generate(prompt, **params)

                    # Error from provider
                    if resp.get("error"):
                        last_error = f"{name} error: {resp['error']}"
                        break

                    # Empty text → treat as fail
                    text = resp.get("text", "")
                    if not text.strip():
                        last_error = f"{name} empty text"
                        break

                    # Too slow → fail
                    latency = resp.get("latency_ms", 0)
                    if latency > TIMEOUT_MS:
                        last_error = f"{name} timeout > {TIMEOUT_MS} ms"
                        break

                    # SUCCESS
                    return resp

                except Exception as e:
                    last_error = f"{name} exception: {e}"
                    continue

            # After retries → go to next provider
            continue

        return {
            "text": "",
            "raw": None,
            "model": "none",
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_ms": 0,
            "error": last_error or "All providers failed",
        }
