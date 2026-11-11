"""
Feature Schema Adapter & Ingestion Preprocessing.

Resolves the 63-feature vs 78-feature incompatibility between cleaned training sets
and the 78-dimension Deep Autoencoder architecture.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


class FeatureSchemaAdapter:
    """
    Adapts incoming feature vectors between 63-feature cleaned datasets and
    78-feature Autoencoder tensors.
    """

    def __init__(self, canonical_78_features: Optional[List[str]] = None) -> None:
        self.canonical_features: List[str] = canonical_78_features or self._default_canonical_features()
        self.input_dim: int = 78
        self.cleaned_dim: int = 63

    @staticmethod
    def _default_canonical_features() -> List[str]:
        """Provides default ordered column names for CICIDS2017 78 numerical features."""
        return [f"feature_{i}" for i in range(1, 79)]

    def align_to_78_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Takes an incoming DataFrame (either 63 or 78 features) and returns a clean
        NumPy array of shape (N, 78) with constant columns imputed.
        """
        n_cols = df.shape[1]
        if n_cols == 78:
            return df.to_numpy(dtype=np.float32)
        elif n_cols == 63:
            # Impute missing constant columns with 0.0 padding
            arr = np.zeros((df.shape[0], 78), dtype=np.float32)
            arr[:, :63] = df.to_numpy(dtype=np.float32)
            return arr
        else:
            raise ValueError(f"Unexpected feature count: {n_cols}. Expected 63 or 78.")

    def sanitize_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """Replaces NaNs and infinite values resulting from zero-division in network flows."""
        return np.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)

