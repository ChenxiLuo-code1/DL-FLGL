"""Data I/O only; all scaling is fitted inside each training fold."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy import sparse
from sklearn.datasets import make_multilabel_classification

def dense(value):
    return np.asarray(value.toarray() if sparse.issparse(value) else value, dtype=float)

def load_dataset(path):
    path = Path(path)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as arrays:
            x, y = dense(arrays['X']), dense(arrays['Y'])
    elif path.suffix.lower() == '.mat':
        arrays = loadmat(path)
        x = next((dense(arrays[k]) for k in ('data', 'X', 'features') if k in arrays), None)
        y = next((dense(arrays[k]) for k in ('target', 'Y', 'labels') if k in arrays), None)
        if x is None or y is None:
            raise ValueError("MAT requires data/X/features and target/Y/labels")
    else:
        raise ValueError("Supported formats: .mat and .npz")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("X and Y must be matrices")
    if len(y) != len(x):
        y = y.T
    y = np.where(y == -1, 0, y)
    if len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid data dimensions, nonfinite entries or nonlogical labels")
    return x, y

def demo_data(seed=42):
    x, y = make_multilabel_classification(n_samples=100, n_features=8, n_classes=4, n_labels=2,
                                          allow_unlabeled=False, random_state=seed)
    return x.astype(float), y.astype(float)
