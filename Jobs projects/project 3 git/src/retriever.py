"""
retriever.py
-  Recovery Functions: filter  by product and returns chunks top-k.
"""
#comment to test overwritting

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from .embed_index import VectorIndex, build_embeddings, load_embedding_model
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    """
    - Receives a corpus (DF wiht columns: 'chunk' and 'product').
    - Creates embbedings and an index VectorIndex(cosine).
    - query() returns the chunks top-k; opcionally filters by product.
    """
    def __init__(self, corpus: pd.DataFrame, model_name: str="sentence-transformers/all-MiniLM-L6-v2",) -> None:
        if not {"chunk", "product"}.issubset(corpus.columns):
            raise ValueError("Corpus must contain 'chunk' and 'product' columns")
        self.corpus = corpus.reset_index(drop=True)
        self.model: SentenceTransformer = load_embedding_model(model_name)    
        
        #corpus embeddings (sting lists)
        chunks: List[str] = self.corpus["chunk"].astype(str).tolist() #lost of size n
        self.embeddings = build_embeddings(chunks, self.model) #(n,d) float32 unitary
        #index over all corpus
        self.index=VectorIndex(self.embeddings) #index over all corpus

    def query(self, question: str, product: Optional[str]=None, top_k: int=6, return_scores: bool=False):
        # query embedding(1,d)
        qv= build_embeddings([question], self.model)  #(n,d) float32 unitary
        
        #if they ask filter by product
        if product:
            mask = self.corpus["product"].eq(product).to_numpy()  #shape(n,)
            if mask.any():
                vecs=self.embeddings[mask]                        #(m,d)
                sims=cosine_similarity(qv, vecs)[0]               #(m,)
                k = min(top_k, vecs.shape[0])
                order = np.argsort(-sims)[:k]                     #local indices 0..m-1
                chunks = self.corpus.loc[mask, "chunk"].iloc[order].tolist()
                return (chunks, sims[order].tolist()) if return_scores else chunks
            # if no rows for that product, fall back to full corpus

        #with no filters: entire index using
        k=min(top_k, len(self.corpus))
        idxs, scores= self.index.search(qv, top_k=top_k)
        chunks= self.corpus["chunk"].iloc[idxs].tolist()
        return (chunks, scores.tolist()) if return_scores else chunks #return chunk with False or chunk and score with True
