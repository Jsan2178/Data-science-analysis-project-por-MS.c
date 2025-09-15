# Telco Customer Churn (ROI-aware)

**Data:** Kaggle – Telco Customer Churn  
**Model:** `Pipeline(StandardScaler → MLPClassifier)` (binary classification)

**Test results**
- ROC-AUC ≈ **0.85**
- PR-AUC (Average Precision) ≈ **0.64** (base rate ≈ **0.26**)
- At threshold **t = 0.45** → Precision **0.48**, Recall **0.87**
- ROI cutoff **t\* = C/(r·CLV) ≈ 0.20** → Precision **0.39**, Recall **0.97**  
  → Using t=0.45 skips **~253** customers with **positive expected value (EV>0)**

**Decision rule**
- EV = *p*·*r*·*CLV* − *C* → intervene if *p* ≥ **t\***, or act on **Top-K by EV** when capacity is limited.

**How to use**
- Open `telco_churn.ipynb`, run all cells, adjust threshold **t** to trade precision vs. recall or set **t\*** for ROI.
