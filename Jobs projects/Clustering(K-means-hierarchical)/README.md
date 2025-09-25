# Customer Segmentation — K-Means & Hierarchical (Ward)

**Summary:** Mini demo segmenting `Mall_Customers.csv` We compare cluster counts using the K-Means elbow curve (plotting inertia vs. k) and a Ward dendrogram, selecting k via the largest vertical gap.
You can see the performance by clicking in the notebook **k_means_clustering.ipynb** here.

## Data
- `Mall_Customers.csv` (gender, age, annual income, spending score)

## Methods
- K-Means with `n_clusters=5` and 2D visualization
- SciPy dendrogram with `method="ward"` (choose `k` via the **largest vertical gap** rule)

## Run
```bash
pip install numpy pandas matplotlib scipy scikit-learn
jupyter lab k_means_clustering.ipynb
```

Notes

- Scale features if needed (StandardScaler).

- Optional k validation: Silhouette, Calinski–Harabasz, Davies–Bouldin (e.g., test k=3..6).

Files

- k_means_clustering.ipynb — K-Means (k=5) + Ward dendrogram

- Mall_Customers.csv — dataset
