"""
aspects.py
- Classifies reviews by simple aspects using keywords and asigns polarity per rating (heuristic).
"""

#comment to test overwritting
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List

DEFAULT_ASPECTS: Dict[str, List[str]] = {
    "battery": ["battery", "batería", "bateria", "charge", "duración", "dura", "power"],
    "price":   ["price", "precio", "cost", "value", "vale", "caro", "barato"],
    "quality": ["quality", "calidad", "build", "material", "acabado"],
    "durability": ["durable", "durabilidad", "last", "romp", "months", "años"],
    "shipping": ["shipping", "envío", "enviado", "paquetería", "llegó", "entrega"]
}

def tag_aspects(corpus: pd.DataFrame, aspects: Dict[str, List[str]] = None) -> pd.DataFrame:
    """Add boolean columns per aspect if the chunk mentions some keyword."""
    if aspects is None:
        aspects = DEFAULT_ASPECTS
    df = corpus.copy()                         #copy original DF
    text = df["chunk"].str.lower().fillna("")  #take text and convert to lowercase and fill NaN with ""
    for a, kws in aspects.items():             
        df[a] = text.apply(lambda t: any(kw in t for kw in kws)) #.apply applies function lambda 
    return df                                                    # returns True if a keyword appears like subchain in the review

def sentiment_from_rating(rating: float) -> int:
    """Heuristic: >=4 positive (+1), <=2 negative (-1), 3 neutral (0), NaN -> 0."""
    try:
        r = float(rating)
    except Exception:
        return 0
    if r >= 4:
        return 1
    if r <= 2:
        return -1
    return 0

#return how many reviews contains the aspect 
def aggregate_aspects(df_tagged: pd.DataFrame, product: str = None) -> pd.DataFrame:
    """Adds polarity per aspect (mean) and coverage (% of chunks tagged)."""
    df = df_tagged.copy()
    if product is not None:
        df = df[df["product"] == product]
    df["pol"] = df["rating"].apply(sentiment_from_rating)

    rows = []
    #detecting which columns are aspects: must exists as keys on DEFAULT_ASPECTS
    aspect_cols = [c for c in df.columns if c in DEFAULT_ASPECTS.keys()] 
    for a in aspect_cols:
        sub = df[df[a]] #just the chunks where the aspect a==True (mentioned)
        if len(sub) == 0:
            rows.append({"aspect": a, "coverage": 0.0, "mean_polarity": 0.0, "n": 0})
        else:
            rows.append({
                "aspect": a,                                              #aspect name
                "coverage": round(len(sub) / max(1, len(df)) * 100, 2),   # % of coverage
                "mean_polarity": round(sub["pol"].mean(), 3),             # average sentiment
                "n": len(sub)                                             #How many chunks mentioned it 
            })
    return pd.DataFrame(rows).sort_values(by="coverage", ascending=False).reset_index(drop=True)
