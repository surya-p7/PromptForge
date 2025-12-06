from .base import BaseLLMProvider
import time

class MockProvider(BaseLLMProvider):
    def generate(self, prompt: str, **params):
        t = time.time()
        return {"text": f"[MOCK] {prompt}", "raw": None, "model": "mock", "tokens_in":0, "tokens_out":0, "latency_ms":1, "error": None}

    def embed(self, text: str, **params):
        return {"embedding":[0.01]*8, "model":"mock", "latency_ms":1, "error":None}
