# XGBoost Breast Cancer Classifier (JupyterLab)

A JupyterLab project that trains an **XGBoost (XGBClassifier)** model to classify tumors as **benign (0)** or **malignant (1)** using a breast-cancer cytology dataset (loaded from `Data.csv`).

## What this project does
- Loads `Data.csv`
- Cleans/prepares the data:
  - Drops the ID column: `Sample code number`
  - Converts the original label to binary:
    - `Class == 4` → `1` (malignant / positive class)
    - otherwise → `0` (benign)
- Runs quick EDA (summary stats + distribution plots)
- Trains a baseline **XGBoost** model
- Evaluates with:
  - confusion matrix + classification report (test set)
  - **10-fold cross-validation**
- Tries hyperparameter tuning with:
  - `GridSearchCV` (5-fold CV)
  - `RandomizedSearchCV` (10-fold CV)
- Experiments with a custom probability threshold using `predict_proba`

## Expected dataset columns
Your notebook assumes `Data.csv` includes these columns:

- `Sample code number` (dropped)
- `Clump Thickness`
- `Uniformity of Cell Size`
- `Uniformity of Cell Shape`
- `Marginal Adhesion`
- `Single Epithelial Cell Size`
- `Bare Nuclei`
- `Bland Chromatin`
- `Normal Nucleoli`
- `Mitoses`
- `Class` (label)

## Results (from the notebook)
Baseline XGBoost (test set):
- Confusion matrix: `[[85, 2], [1, 49]]`
- Accuracy: **97.81%**

10-fold cross-validation (on training split):
- Mean accuracy: **97.25%**
- Std: **1.88%**

Hyperparameter search (CV best scores):
- GridSearchCV (972 candidates, 5-fold): **97.62%**
- RandomizedSearchCV (500 candidates, 10-fold): **97.26%**

Threshold experiment example (`threshold = 0.7` on tuned model):
- Confusion matrix: `[[84, 3], [2, 48]]`

## Tech stack
- Python (JupyterLab)
- pandas, numpy, matplotlib
- scikit-learn
- xgboost
- scipy (for randomized parameter distributions)

## How to run
1. Clone/download this project.
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn xgboost scipy
