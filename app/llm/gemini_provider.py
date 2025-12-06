import time, os
from .base import BaseLLMProvider
from ..config import settings
try:
    # import google.generativeai as genai
    import google.genai as genai
    client_class = genai.Client
except Exception:
    genai = None
    client_class = None # Set the class to None if the import fails


class GeminiProvider(BaseLLMProvider):
    def __init__(self):
        # We don't need genai.configure() here if we use the Client constructor 
        # or rely on the SDK to find the GEMINI_API_KEY environment variable.
        if client_class:
            self.client = client_class() # Use the acquired class
        else:
            raise ImportError("The 'Client' class could not be imported from the Gemini SDK.")
        self.model_name = os.getenv("GEMINI_MODEL", settings.DEFAULT_MODEL)
        # self.client = Client() # The Client auto-detects GEMINI_API_KEY from .env/env

    def generate(self, prompt: str, **params):
        start = time.time()
        try:
            # 1. Use client.models.generate_content (or client.generate_content)
            # The SDK will convert the prompt string into the required contents format.
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt, # Use 'contents' parameter
                config={
                    "temperature": params.get("temperature", 0.0)
                }
            )

            # 2. Extract text using the correct attribute name (.text)
            text = resp.text 
            
            # The response object also contains token counts for production usage
            # Token counting requires a specific call, but we'll use 0 for now
            tokens_in = resp.usage_metadata.prompt_token_count if hasattr(resp, 'usage_metadata') else 0
            tokens_out = resp.usage_metadata.candidates_token_count if hasattr(resp, 'usage_metadata') else 0
            
            latency = (time.time() - start) * 1000
            
            return {
                "text": text,
                "raw": resp.model_dump(), # Use model_dump for structured Pydantic output
                "model": self.model_name,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": latency,
                "error": None
            }
        except Exception as e:
            return {"text": "", "raw": None, "model": self.model_name, "tokens_in": 0, "tokens_out": 0, "latency_ms": 0, "error": str(e)}
    def embed(self, text: str, **params):
        return {"embedding": [], "model": self.model_name, "latency_ms":0, "error":"embeddings not implemented"}
