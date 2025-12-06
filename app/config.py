import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    ENV = os.getenv("ENV", "dev")
    LLM_BACKEND = os.getenv("LLM_BACKEND", "mock").lower()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini-1.5")
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", 5))
    TOKEN_LIMIT = int(os.getenv("TOKEN_LIMIT", 3200))

settings = Settings()
