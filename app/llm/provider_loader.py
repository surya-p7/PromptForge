# app/llm/provider_loader.py
from typing import List, Tuple, Dict, Any
from .base import BaseLLMProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider
from .mock_provider import MockProvider

import os
_PROVIDER_CLASSES = {
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
    "openai": OpenAIProvider,
    "mock": MockProvider,
}


def _try_init_provider(name: str, cls) -> Tuple[BaseLLMProvider | None, str | None]:
    """
    Try to instantiate a provider. Never raise.
    Returns (provider_instance or None, error_message or None).
    """
    try:
        provider = cls()
        return provider, None
    except Exception as e:
        return None, str(e)


def discover_providers(
    preferred_order: List[str] | None = None,
) -> Tuple[List[BaseLLMProvider], Dict[str, Dict[str, Any]]]:
    """
    Auto-detect which providers are actually usable.

    Returns:
      - providers: list of successfully created provider instances in priority order
      - report: dict[name] = {"ok": bool, "error": str | None}
    """
    if preferred_order is None:
        # preferred_order = ["gemini", "ollama", "openai", "mock"]
        enable_ollama = os.getenv("ENABLE_OLLAMA", "false").lower() == "true"

        preferred_order = ["gemini"]
        if enable_ollama:
            preferred_order.append("ollama")
        preferred_order.extend(["openai", "mock"])

    providers: List[BaseLLMProvider] = []
    report: Dict[str, Dict[str, Any]] = {}

    for name in preferred_order:
        cls = _PROVIDER_CLASSES[name]
        provider, err = _try_init_provider(name, cls)
        if provider is not None:
            providers.append(provider)
            report[name] = {"ok": True, "error": None}
        else:
            report[name] = {"ok": False, "error": err}

    return providers, report


def build_provider_chain() -> Tuple[List[BaseLLMProvider], Dict[str, Dict[str, Any]]]:
    """
    Build the provider chain and print a small startup report.
    If all real providers fail, fall back to MockProvider only.
    """
    providers, report = discover_providers()

    print("\n[Provider Discovery]")
    for name, info in report.items():
        status = "✓" if info["ok"] else "✗"
        msg = info["error"] or "ready"
        print(f"  {status} {name}: {msg}")

    if not providers:
        # Absolute last resort: MockProvider should always work
        mock = MockProvider()
        providers = [mock]
        print("  ! No real providers available, using MockProvider only.\n")
    else:
        chain_names = [p.__class__.__name__ for p in providers]
        print(f"  → Active chain: {', '.join(chain_names)}\n")

    return providers, report


def load_provider() -> BaseLLMProvider:
    """
    Backwards-compatible helper:
    Return the first available provider from the discovered chain.
    Used by ReversePromptService etc.
    """
    providers, _ = build_provider_chain()
    return providers[0]
