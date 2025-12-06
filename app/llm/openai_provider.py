import time, os
from openai import OpenAI
from .base import BaseLLMProvider
from ..config import settings

class OpenAIProvider(BaseLLMProvider):
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, prompt: str, **params):
        start=time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role":"user","content":prompt}],
                temperature=params.get("temperature",0.0)
            )
            msg = resp.choices[0].message.content
            latency=(time.time()-start)*1000
            usage = getattr(resp, "usage", None)
            tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
            tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0
            return {"text": msg, "raw": resp, "model": self.model_name, "tokens_in": tokens_in, "tokens_out": tokens_out, "latency_ms":latency, "error":None}
        except Exception as e:
            return {"text":"", "raw":None, "model":self.model_name, "tokens_in":0, "tokens_out":0, "latency_ms":0, "error":str(e)}

    def embed(self, text: str, **params):
        try:
            resp = self.client.embeddings.create(model="text-embedding-3-small", input=text)
            emb = resp.data[0].embedding
            return {"embedding": emb, "model":"text-embedding-3-small", "latency_ms":0, "error": None}
        except Exception as e:
            return {"embedding": [], "model":"", "latency_ms":0, "error": str(e)}
