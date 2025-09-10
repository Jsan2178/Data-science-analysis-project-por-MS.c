# Project: CIFAR-10 CNN (PyTorch) – Baseline and Improvements

- **Dataset:** CIFAR-10 (32×32 RGB, 10 classes)  
- **Device:** CPU  
- **Frameworks:** PyTorch, scikit-learn (for metrics/plots)

---

## Baseline

**Architecture**
- Conv2d(3→6, k=5) → ReLU → MaxPool(2)
- Conv2d(6→16, k=5) → ReLU → MaxPool(2)
- Flatten (16×5×5)
- Linear(16×5×5 → 120) → ReLU
- Linear(120 → 84) → ReLU
- Linear(84 → 10) *(logits)*

**Training setup**
- Optimizer: SGD (momentum=0.9)  
- Learning rate: **1e-3**  
- Batch size: **32**  
- Epochs: **6**

**Result**
- **Test accuracy:** **57.8%**  
- **Per-class accuracy (sample):** plane 57.8, car 60.2, bird 38.3, cat 39.7, deer 52.9, dog 51.6, frog 77.6, horse 64.3, ship 63.2, truck 72.0 (%)

---

## Improved setup

**Architecture changes**
- Kept the two conv blocks (same as baseline).
- Expanded the dense head:
  - Linear(16×5×5 → **128**) → ReLU  
  - Linear(**128 → 128**) → ReLU  
  - Linear(**128 → 128**) → ReLU  
  - Linear(**128 → 10**) *(logits)*
- Added `Dropout(p=0.2)` and `Dropout2d(p=0.2)` modules (declared; optional in the forward).

**Training changes**
- Optimizer: **Adam** (`torch.optim.Adam`)  
- Learning rate: **5e-4**  
- Weight decay: **1e-4**  
- Batch size: **64**  
- Epochs: **14**  
- Reproducibility: set **seed=42** (Python, NumPy, PyTorch; cuDNN deterministic flags).

**Result**
- **Test accuracy:** **63.6%** (**+5.8 pp vs. baseline**)  
- **Per-class accuracy:** plane 63.9, car 74.4, bird 46.6, cat 39.9, deer 60.2, dog 57.3, frog 69.8, horse 76.2, ship 76.2, truck 71.3 (%)

---

## Evaluation & Logging

- Computed overall accuracy, per-class accuracy, and a confusion matrix using `sklearn.metrics`:
  - `confusion_matrix`, `ConfusionMatrixDisplay`, `classification_report`.
- Saved checkpoints and weights:
  - **Save:** `torch.save(model.state_dict(), "cnn.pth")`
  - **Load:** instantiate the same `ConvNet()`, then:
    ```pyth
