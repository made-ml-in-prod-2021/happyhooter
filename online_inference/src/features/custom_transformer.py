import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MyStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray) -> "MyStandardScaler":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = (x - self.mean) / self.std
        return x
