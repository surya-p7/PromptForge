# app/services/dynamic_prompting.py
from ..prompts import assemble_prompt
from ..llm.provider_loader import load_provider
from ..llm.provider_router import ProviderRouter
from ..config import settings

# optional provider imports for override
from ..llm.gemini_provider import GeminiProvider
from ..llm.ollama_provider import OllamaProvider
from ..llm.openai_provider import OpenAIProvider
from ..llm.mock_provider import MockProvider


class DynamicPromptService:
    def __init__(self, provider=None, retriever=None):
        # Keep provider for backwards compatibility
        self.provider = provider or load_provider()
        self.retriever = retriever

    def answer(self, final_prompt: str, user_query: str, metadata: dict = None):
        metadata = metadata or {}

        # ---------------------------
        # 1. Determine provider override
        # ---------------------------
        provider_override = metadata.get("provider_override", "auto")

        if provider_override == "auto":
            router = ProviderRouter()       # auto discovery chain
        else:
            provider_map = {
                "gemini": GeminiProvider,
                "ollama": OllamaProvider,
                "openai": OpenAIProvider,
                "mock": MockProvider,
            }

            provider_cls = provider_map.get(provider_override)
            if provider_cls:
                router = ProviderRouter([provider_cls()])
            else:
                router = ProviderRouter()   # fallback to auto

        # ---------------------------
        # 2. Retrieve chunks (if RAG enabled)
        # ---------------------------
        chunks = []
        if self.retriever:
            chunks = self.retriever.retrieve(
                user_query, top_k=settings.RETRIEVAL_TOP_K
            )

        # ---------------------------
        # 3. Assemble final prompt
        # ---------------------------
        assembled = assemble_prompt(
            final_prompt,
            chunks,
            user_query,
            constraints="Deterministic, temperature=0, cite sources inline",
        )

        # ---------------------------
        # 4. Generate final answer
        # ---------------------------
        return router.safe_generate(assembled, temperature=0.0)
