# Amazon Reviews — RAG-lite Q&A + Aspect Sentiment

A **RAG-lite** assistant over Amazon reviews: performs **semantic retrieval by product** (Sentence-Transformers embeddings), answers a **question with evidence** (top-k review chunks), and shows an **aspect summary** (price, battery, quality, durability, shipping).  
Includes **retrieval metrics** (Recall@k / MRR@k) and a reproducible demo notebook.

---

## Project Structure

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

## How to Run

```bash
pip install -r requirements.txt

---

## Requirements

```txt
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
sentence-transformers>=2.4
transformers>=4.40
torch>=2.1
tqdm>=4.66
ipywidgets>=8.0

## How to Run

```bash
pip install -r requirements.txt

# Use the CSV in ./data/1429_1.csv (or sample_reviews.csv)
# Or place the full dataset as ./data/amazon_full.csv (auto-detected)
jupyter notebook Notebooks/01_demo_qa_aspects.ipynb


> Important: You must have **three backticks** before and after each block. If any are missing, GitHub will render it as regular text and spacing will collapse.

---

## Option 2 — Super compact, “single-flow” section
If you want it to look like one continuous “single line per command/tree row”, wrap the whole chunk in **one** code block:

```markdown
## Quickstart

Project3/
├─ Notebooks/01_demo_qa_aspects.ipynb
├─ src/{dataio.py, embed_index.py, retriever.py, qa.py}
├─ data/1429_1.csv # or sample_reviews.csv
├─ artifacts/ # generated; do not commit
├─ requirements.txt
└─ README.md


### Place data

Use the CSV included in ./data/1429_1.csv or

Download the full dataset and save it as ./data/amazon_full.csv.
The notebook auto-detects amazon_full.csv; if not present, it falls back to the small file.

### Open the notebook

jupyter notebook Notebooks/01_demo_qa_aspects.ipynb