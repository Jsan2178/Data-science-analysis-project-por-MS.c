# Grid Search with SVM (Scikit-learn)

This project demonstrates a classic **SVM classification** workflow using `scikit-learn`, including **scaling**, **train/test split**, **k-fold cross-validation**, and **hyperparameter tuning with GridSearchCV** (exploring `C`, `kernel`, and `gamma`) on the `Social_Network_Ads.csv` dataset.

> **Main notebook:** `grid_search.ipynb`

---

## 🧠 Goal
Train an **SVM** for a binary task and **optimize hyperparameters** via `GridSearchCV`, comparing `linear` and `rbf` kernels, and evaluating with **accuracy**, **ROC-AUC**, **PR-AUC**, **confusion matrix**, and **classification report**.

## 📁 Project structure
```
├── grid_search.ipynb        # Full workflow
├── Social_Network_Ads.csv   # Dataset (not included by default)
└── README.md                # This document
```

## 🧩 Dataset
- **Expected file:** `Social_Network_Ads.csv` at the project root.
- **Target:** last column (binary).
- **Features:** all columns except the last.
- If you swap in your own CSV, keep the same loading interface:
  ```python
  X = dataset.iloc[:, :-1].values
  y = dataset.iloc[:, -1].values
  ```

## ⚙️ Environment & requirements
Install minimal deps:
```bash
pip install numpy pandas scikit-learn matplotlib
```
Notebook uses: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

## ▶️ How to run
1. Clone/download the repo and place `Social_Network_Ads.csv` at the root.
2. Open the notebook:
   ```bash
   jupyter notebook grid_search.ipynb
   ```
3. Run all cells in order.

> **Note:** The flow includes `train_test_split(random_state=0)` and `StandardScaler`. If you prefer scaling inside a `Pipeline`, adjust grid keys to `clf__C`, `clf__kernel`, `clf__gamma`.

## 🔧 Model & hyperparameter search
- **Base model:** `SVC(kernel='rbf', probability=True, random_state=0)`
- **Example grid** (randomly generated in the notebook):
  - `C`: 20 values uniform in `[0.1, 10]`
  - `gamma`: 50 values uniform in `[0.1, 10]`
  - `kernel`: `['linear']` and `['rbf']`
- **Validation:** `cv=10`, `scoring='accuracy'`, `n_jobs=-1`

Pseudocode (simplified):
```python
c_values = np.random.uniform(0.1, 10, 20)
gamma_values = np.random.uniform(0.1, 10, 50)
param_grid = [
  {'C': c_values, 'kernel': ['linear']},
  {'C': c_values, 'kernel': ['rbf'], 'gamma': gamma_values}
]

grid = GridSearchCV(estimator=classifier,
                    param_grid=param_grid,
                    scoring='accuracy',
                    cv=10,
                    n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_params = grid.best_params_
best_cv_acc = grid.best_score_
```

## 📊 Metrics & visuals
The notebook prints and/or plots:
- **Confusion matrix**, **accuracy**, **classification_report**
- **ROC-AUC** and **PR-AUC** (requires `probability=True`)
- **Decision boundaries** for train/test (2D)

### Example (fill in with your results)
- `Best params:` `{'C': …, 'kernel': 'rbf', 'gamma': …}`
- `Best CV accuracy:` `…`
- `Test Accuracy:` `…`
- `ROC-AUC:` `…` &nbsp; `PR-AUC:` `…`
- **Confusion matrix (test):**
  ```
  [[TN, FP],
   [FN, TP]]
  ```

## 🔁 Reproducibility
- Fixed `random_state=0` in `train_test_split` and `SVC`.
- **Random grids** (`np.random.uniform`) will change on re-runs; set a seed (`np.random.seed(...)`) to reproduce the exact grid.

## 🧪 Suggested extensions
- Wrap everything in a `Pipeline(StandardScaler(), SVC())` to avoid leakage.
- Try explicit `StratifiedKFold` and extra metrics (`f1`, `balanced_accuracy`).
- Use `RepeatedStratifiedKFold` or **Bayesian Optimization** (`skopt`, `optuna`).
- Save the best model with `joblib` and provide a simple `predict.py` CLI.

## 📝 License
Educational purposes only. Adjust license as needed.

---

## 📣 Portfolio blurb (copy-ready)
> SVM classifier with hyperparameter optimization via GridSearchCV (scikit-learn), 10-fold stratified CV, and a full report of metrics (accuracy, ROC-AUC, PR-AUC), including decision-boundary visualizations and best-found parameters over `C`, `kernel`, and `gamma`.
