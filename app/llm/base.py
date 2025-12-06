from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **params) -> Dict[str, Any]:
        """
        Return standardized result:
        {
          "text": str,
          "raw": Any,
          "model": str,
          "tokens_in": int,
          "tokens_out": int,
          "latency_ms": float,
          "error": str | None
        }
        """
        pass

    @abstractmethod
    def embed(self, text: str, **params) -> Dict[str, Any]:
        """
        Return:
        {
          "embedding": List[float],
          "model": str,
          "latency_ms": float,
          "error": str | None
        }
        """
        pass
