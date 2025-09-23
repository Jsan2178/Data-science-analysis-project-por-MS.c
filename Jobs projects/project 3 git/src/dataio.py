#comment to test overwritting

import pandas as pd, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer #some libraries
def load_reviews(csv_path):
    df=pd.read_csv(csv_path)
    rename_ft = { "reviews.text": "text",
        "reviews.rating": "rating",
        "name": "product"
    }
    df = df.rename(columns={k: v for k, v in rename_ft.items() if k in df.columns}) #rename for clarifying data
    required = ["text", "product"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f'Miss column required: {col}')
    df = df.dropna(subset=["text", "product"]).reset_index(drop=True) #this reset index for holes after droping NaN
    return df

#Chunking text in pieces with limited characters
def chunk_text(text, max_chars=500):
    text= str(text)
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

#corpus build 
def build_corpus(df: pd.DataFrame, text_col: str = "text", product_col: str = "product", rating_col: str = "rating", max_chars: int = 500) -> pd.DataFrame:
    rows=[]
    for _, r in df.iterrows():
        chunks = chunk_text(r[text_col], max_chars=max_chars)
        for ch in chunks:
            rows.append({
                "product": r[product_col],
                "rating": r.get(rating_col, np.nan), #place nan if raintg_col doesnt exist
                "chunk": ch #create rows for each ch with same product and rating(as it should be) 
            })                       #because it comes from the same text review and product and rating
    return pd.DataFrame(rows)
