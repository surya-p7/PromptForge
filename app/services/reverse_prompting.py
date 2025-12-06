# app/services/reverse_prompting.py
from ..prompts import build_reverse_prompt
from ..llm.provider_loader import load_provider
import json


class ReversePromptService:
    def __init__(self, provider=None):
        self.provider = provider or load_provider()

    def generate_meta(self, user_query: str):
        """
        Generate a cleaned, clarified, minimal final prompt.
        No clarifying questions.
        No unnecessary expansion.
        Direct transformation only.
        """

        # Build the reverse prompt for LLM
        prompt = build_reverse_prompt(user_query)

        # Call the provider (Gemini / Ollama / OpenAI / Mock)
        resp = self.provider.generate(prompt, temperature=0.0)
        text = resp.get("text", "").strip()

        # Try strict JSON from provider
        try:
            parsed = json.loads(text)
            if "final_prompt" not in parsed:
                # Ensure minimal structure
                parsed = {
                    "final_prompt": text,
                    "context_requirements": []
                }
            return parsed

        except Exception:
            # Fallback if provider returns non-JSON text
            return {
                "final_prompt": text,
                "context_requirements": []
            }
