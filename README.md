# LLM Orchestration Engine 

Goals:
- Reverse prompting, dynamic prompts
- RAG-ready retriever (FAISS + sentence-transformers)
- Multi-provider adapter (Gemini / OpenAI / Ollama / Mock)
- Dockerized, CI, observability hooks

Run locally:
1. Copy .env.example -> .env and set values.
2. docker compose up --build
3. POST /api/generate { "text": "Summarize..." }

Switch backend:
export LLM_BACKEND=ollama  # or gemini / openai / mock

Extend:
- Implement persistent vector DB
- Add per-route provider routing, fallback policies
- Add real monitoring (Prometheus / Grafana)



Run docker compose up --build to smoke-test.
Run pytest -q to validate tests.
