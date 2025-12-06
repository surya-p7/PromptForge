from fastapi import FastAPI, HTTPException
from .middleware import RequestTracingMiddleware
from .llm.provider_loader import load_provider
from .services.reverse_prompting import ReversePromptService
from .services.dynamic_prompting import DynamicPromptService
# from .services.rag.retriever import FaissRetriever
from .api_models import GenerateRequest, GenerateResponse
from .logger import request_logger
from .config import settings
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="LLM Orchestration Engine")
app.add_middleware(RequestTracingMiddleware)
# Allow frontend (file:// or http://localhost) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # dev only; later you can restrict
    allow_credentials=True,
    allow_methods=["*"],          # allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

provider = load_provider()

# Initialize retriever lazily (disabled by default to keep tests fast)
retriever = None

reverse_service = ReversePromptService(provider=provider)
dynamic_service = DynamicPromptService(provider=provider, retriever=retriever)

@app.get("/health")
def health():
    return {"status":"ok", "backend": settings.LLM_BACKEND}

@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    rid = None
    log = request_logger(rid, route="/api/generate")
    log(20, "generate_called", body=req.text[:200])
    # 1) reverse prompt meta
    meta = reverse_service.generate_meta(req.text)
    final_prompt = meta.get("final_prompt") or meta.get("prompt") or req.text
    # 2) dynamic assemble + answer
    ans = dynamic_service.answer(final_prompt, req.text, metadata=req.metadata)
    if ans.get("error"):
        raise HTTPException(status_code=500, detail=ans.get("error"))
    return {
        "text": ans.get("text"),
        "model": ans.get("model"),
        "tokens_in": ans.get("tokens_in",0),
        "tokens_out": ans.get("tokens_out",0),
        "latency_ms": ans.get("latency_ms",0),
        "error": ans.get("error"),
        "sources": req.metadata.get("chunks", [])[:settings.RETRIEVAL_TOP_K]
    }
