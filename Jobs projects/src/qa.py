
"""
qa.py
- Generates a simple answer "grounded" from recovered chunks.
- For quick demo an extractive/summarized without heavy LLM is used 
"""
#comment to test overwritting

from __future__ import annotations 
from typing import List

def build_answer(question: str, chunks: List[str], max_refers: int=6) -> str:
    context = "\n\n".join(f"- {c}" for c in chunks[:max_refers])
    answer= f"question: {question}\n\nAnswer (based on recovered reviews):\n{context}"
    return {"answer": answer, "context": context}
