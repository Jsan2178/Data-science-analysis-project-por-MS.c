# NLP — Restaurant Reviews Sentiment (BoW + Naive Bayes)

A compact sentiment-analysis demo in a single Jupyter notebook (`natural_language_processing.ipynb`).  
It classifies restaurant reviews as **positive/negative** using a **Bag-of-Words** representation and **Naive Bayes**.

---

## Overview

- **Dataset:** `Restaurant_Reviews.tsv` (1,000 labeled reviews; binary sentiment, tab-separated).  
- **Preprocessing:** keep letters only (regex), lowercase, tokenization, **stop-word removal** (NLTK) **keeping “not”**, **stemming** (Porter).  
- **Vectorizer:** `CountVectorizer(max_features=1500)`.  
- **Split:** `train_test_split(test_size=0.2, random_state=0)`.  
- **Models:** `GaussianNB` (baseline) and `MultinomialNB` (with simple alpha search).  
- **Metrics:** accuracy, confusion matrix, and **classification report** (precision/recall/F1).

---

## Project Structure

```
.
├─ natural_language_processing.ipynb
├─ Restaurant_Reviews.tsv      # place here (tsv, \t separated)
└─ README.md
```

---

## Quickstart

### 1) Environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -U pip
pip install numpy pandas scikit-learn nltk matplotlib jupyterlab
```

### 2) NLTK stopwords (one time)
```python
import nltk
nltk.download("stopwords")
```

### 3) Run
```bash
jupyter lab
```
Open `natural_language_processing.ipynb` and run all cells.  
Ensure `Restaurant_Reviews.tsv` is in the repo root.

---

## Results (new)

### Gaussian Naive Bayes
**Confusion matrix** (rows = true, cols = predicted):  
`[[55, 42], [12, 91]]`  

**Accuracy:** `0.73`

**Classification report**

| class        | precision | recall | f1-score | support |
|:------------ | --------: | -----: | -------: | ------: |
| Yes          | 0.82 | 0.57 | 0.67 | 97 |
| No           | 0.68 | 0.88 | 0.77 | 103 |
| **accuracy** |      |      | **0.73** | 200 |
| macro avg    | 0.75 | 0.73 | 0.72 | 200 |
| weighted avg | 0.75 | 0.73 | 0.72 | 200 |

> Notes: Slight class imbalance (97 vs 103). Macro/weighted averages are reported for fairness.

### Multinomial Naive Bayes (tuned α)
- Simple random search over α ∈ [0.01, 10].  
- **Best observed accuracy:** ≈ **0.79**  
- Example confusion matrix recorded in notebook: `[[74, 23], [19, 84]]`
**Classification report**
  
| class        | precision | recall | f1-score | support |
|:------------ | --------: | -----: | -------: | ------: |
| Yes          | 0.80 | 0.76 | 0.78 | 97 |
| No           | 0.79 | 0.82 | 0.80 | 103 |
| **accuracy** |      |      | **0.79** | 200 |
| macro avg    | 0.79 | 0.79 | 0.79 | 200 |
| weighted avg | 0.79 | 0.79 | 0.79 | 200 |

*(Your exact scores can vary slightly due to randomness and environment.)*

---

## Predict a New Review

```python
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# example review
new_review = "I hate this restaurant, it's not as good as I thought"

# same preprocessing as training
new_review = re.sub(r"[^a-zA-Z]", " ", new_review).lower().split()
ps = PorterStemmer()
sw = stopwords.words("english")
if "not" in sw: sw.remove("not")              # keep negation
new_review = " ".join(ps.stem(w) for w in new_review if w not in set(sw))

X_new = cv.transform([new_review]).toarray()  # cv: fitted CountVectorizer
y_new = clf.predict(X_new)                    # clf: fitted NB model
print(y_new)                                  # 1 = positive, 0 = negative
```

---

## Reproduce the Notebook Steps

1. Load TSV with `pandas.read_csv(..., delimiter="\t", quoting=3)`.  
2. Clean & stem text; retain “not”.  
3. Vectorize with `CountVectorizer(max_features=1500)`.  
4. Split 80/20.  
5. Train **GaussianNB** and **MultinomialNB** (search α).  
6. Evaluate: accuracy, confusion matrix, classification report.

---

## Ideas to Extend

- Try `TfidfVectorizer` and bigrams `(1,2)`.  
- Use `GridSearchCV`/`RandomizedSearchCV` with cross-validation.  
- Add ROC/PR curves, per-class F1, and calibration plots.  
- Persist a `Pipeline` (vectorizer + model) with `joblib` for easy reuse.  
- Ship a tiny Streamlit demo for interactive predictions.

---

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `scikit-learn`, `nltk`, `matplotlib`, `jupyterlab`

---

