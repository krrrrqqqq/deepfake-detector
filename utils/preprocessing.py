"""
Preprocessing utilities shared by train_svm.py, test_celebdf.py, and app.py.

Preprocessing order (must be identical at train and inference time):
    1. scaler.transform(X)   — StandardScaler fitted on training data only
    2. normalize(X)          — L2 normalisation (sklearn.preprocessing.normalize)
"""

import numpy as np
from sklearn.preprocessing import normalize as l2_normalize


def apply_preprocessing(X: np.ndarray, scaler) -> np.ndarray:
    """
    Apply StandardScaler + L2 normalisation.

    Parameters
    ----------
    X      : array of shape (n_samples, n_features)
    scaler : a fitted sklearn StandardScaler

    Returns
    -------
    Preprocessed array of the same shape.
    """
    X = scaler.transform(X)
    X = l2_normalize(X)
    return X
