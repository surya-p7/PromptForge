import time, requests, os
from .base import BaseLLMProvider
from ..config import settings

class OllamaProvider(BaseLLMProvider):
    def __init__(self):
        self.url = settings.OLLAMA_URL
        self.model_name = os.getenv("OLLAMA_MODEL", "mistral")

    def generate(self, prompt: str, **params):
        start=time.time()
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,  # <-- CRITICAL CHANGE: Disables streaming for sync client
                "options": {
                    "temperature": params.get("temperature", 0.0)
                }
            }
            # Use the corrected payload
            r = requests.post(f"{self.url}/api/generate", json=payload, timeout=60)
            # r = requests.post(f"{self.url}/api/generate", json={"model": self.model_name, "prompt": prompt}, timeout=60)
            r.raise_for_status()
            resp = r.json()
            text = resp.get("response") or resp.get("output") or str(resp)
            latency=(time.time()-start)*1000
            return {"text": text, "raw": resp, "model": self.model_name, "tokens_in": resp.get("prompt_eval_count",0), "tokens_out": resp.get("eval_count",0), "latency_ms":latency, "error":None}
        except Exception as e:
            return {"text":"", "raw":None, "model":self.model_name, "tokens_in":0, "tokens_out":0, "latency_ms":0, "error":str(e)}

    def embed(self, text: str, **params):
        return {"embedding": [], "model": self.model_name, "latency_ms":0, "error":"not implemented"}
