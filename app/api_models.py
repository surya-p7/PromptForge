from pydantic import BaseModel
from typing import Dict, Any, List

class GenerateRequest(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}
    options: Dict[str, Any] = {}

class GenerateResponse(BaseModel):
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    error: str | None
    sources: List[Dict] | None = None
