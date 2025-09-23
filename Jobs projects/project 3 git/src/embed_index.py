#comment to test overwritting

import numpy as np 
from typing import List, Tuple 
from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity

def load_embedding_model(name: str="sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(name)
def build_embeddings(texts: List[str], model: SentenceTransformer)-> np.ndarray:
    X= model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(X, dtype="float32") #converts X elements in float32 

class VectorIndex:
    """Simple index for cosine similarity """
    def __init__(self, embeddings: np.ndarray):
        self.X = np.asarray(embeddings, dtype="float32")
        if self.X.ndim !=2:
            raise ValueError("embeddings must be (n,d)")
    def search(self, q: np.ndarray, top_k: int=8)-> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype="float32")
        if q.ndim ==1: q=q[None, :] #if (d,) converts to (1,d)
        sims= cosine_similarity(q, self.X)[0] #cosine for all corpus
        idx = np.argsort(-sims)[:top_k]  #top-k by descendent score (-sims) because argsort order from smallest to largest
        return idx, sims[idx]
