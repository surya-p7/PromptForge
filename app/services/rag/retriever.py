import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from ..llm.provider_loader import load_provider

class FaissRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.metadatas = []

    def build(self, docs: List[Dict]):
        embeddings = [self.embedder.encode(d["text"]).astype("float32") for d in docs]
        dim = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.vstack(embeddings))
        self.metadatas = docs

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        q_emb = self.embedder.encode(query).astype("float32")
        D, I = self.index.search(np.expand_dims(q_emb, axis=0), top_k)
        results=[]
        for idx in I[0]:
            results.append(self.metadatas[idx])
        return results
