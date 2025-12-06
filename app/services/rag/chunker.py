from typing import List, Dict
import re
def simple_chunk(text: str, chunk_size: int = 1000) -> List[Dict]:
    words = text.split()
    chunks=[]
    i=0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({"id": f"chunk_{i}", "text": chunk_text})
        i += chunk_size
    return chunks
