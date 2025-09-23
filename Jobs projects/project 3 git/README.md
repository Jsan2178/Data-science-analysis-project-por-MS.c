# Amazon Reviews — RAG-lite Q&A + Aspect Sentiment

A **RAG-lite** assistant over Amazon reviews: performs **semantic retrieval by product** (Sentence-Transformers embeddings), answers a **question with evidence** (top-k review chunks), and shows an **aspect summary** (price, battery, quality, durability, shipping).  
Includes **retrieval metrics** (Recall@k / MRR@k) and a reproducible demo notebook.

---

## Project Structure



```markdown
Project3/
├─ Notebooks/
│ └─ 01_demo_qa_aspects.ipynb
├─ src/
│ ├─ dataio.py
│ ├─ embed_index.py
│ ├─ retriever.py
│ └─ qa.py
├─ data/
│ └─ 1429_1.csv # or sample_reviews.csv
├─ artifacts/ # generated; do not commit
├─ requirements.txt
└─ README.md


```
## Data

1) Create a `data/` folder at the repo root.
2) Download the dataset from this link and place it in `./data/`:
   - **Amazon reviews (full)**: <https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products>
   

The notebook auto-detects:
- If `data/amazon_full.csv` exists, it uses that.

## Requirements

---

```markdown
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
sentence-transformers>=2.4
transformers>=4.40
torch>=2.1
tqdm>=4.66
ipywidgets>=8.0


```
## How to Run
---

```markdown
pip install -r requirements.txt
jupyter notebook Notebooks/01_demo_qa_aspects.ipynb


```

## Quickstart
 

```markdown
Project3/
├─ Notebooks/01_demo_qa_aspects.ipynb
├─ src/{dataio.py, embed_index.py, retriever.py, qa.py}
├─ data/1429_1.csv  # or sample_reviews.csv (or amazon_full.csv)
├─ artifacts/       # generated; do not commit
├─ requirements.txt
└─ README.md


```
### Run
```
pip install -r requirements.txt
jupyter notebook Notebooks/01_demo_qa_aspects.ipynb

```
## Quickstart

```markdown
Project3/
├─ Notebooks/01_demo_qa_aspects.ipynb
├─ src/{dataio.py, embed_index.py, retriever.py, qa.py}
├─ data/1429_1.csv # or sample_reviews.csv
├─ artifacts/ # generated; do not commit
├─ requirements.txt
└─ README.md
```
### Place data

Use the CSV included in ./data/1429_1.csv or

Download the full dataset and save it as ./data/1429_1.csv.
The notebook auto-detects amazon_full.csv; if not present, it falls back to the small file.
```

```
### Open the notebook
```
jupyter notebook Notebooks/01_demo_qa_aspects.ipynb
