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
- Conv2d(3→32, k=5) → ReLU → MaxPool(2)
- Conv2d(32→64, k=3) → ReLU → MaxPool(2)
- Flatten (64×6×6)
- Expanded the dense head:
  - Linear(64×6×6 → **128**) → ReLU  
  - Linear(**128 → 128**) → ReLU  
  - Linear(**128 → 128**) → ReLU  
  - Linear(**128 → 10**) *(logits)*
- Added `Dropout(p=0.2)` and `Dropout2d(p=0.2)` modules (declared; optional in the forward).

**Training changes**
- Optimizer: **Adam** (`torch.optim.Adam`) (weight_decay=1e-3)  
- Learning rate: **5e-4**  
- Weight decay: **1e-4**  
- Batch size: **64**  
- Epochs: **14**  
- Reproducibility: set **seed=42** (Python, NumPy, PyTorch; cuDNN deterministic flags).

**Result**
- **Test accuracy:** **70.5%** (**+12.7 pp vs. baseline**)  
- **Per-class accuracy:** plane 71.9, car 79.5, bird 64.3, cat 42.7, deer 74.1, dog 67.1, frog 79.0, horse 62.0, ship 87.9, truck 76.7 (%)

---

## Evaluation & Logging

- Computed overall accuracy, per-class accuracy, and a confusion matrix using `sklearn.metrics`:
  - `confusion_matrix`, `ConfusionMatrixDisplay`, `classification_report`.
- Saved checkpoints and weights:
  - **Save:** `torch.save(model.state_dict(), "cnn.pth")`
  - **Load:** instantiate the same `ConvNet()`, then:
    ```python
    model.load_state_dict(torch.load("cnn.pth", map_location=device))
    model.eval()
    ```

---

## Notes

- Results are reproducible with the provided seed on the same environment.
- The notebook includes a quick **Load & Inference** section to run predictions without retraining.
