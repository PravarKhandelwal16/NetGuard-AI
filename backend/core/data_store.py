"""
Data Store and Cache Manager for NETGUARD AI.
Handles dataset ingestion, auditing, preprocessing, and active session caching.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SAMPLES_DIR = BASE_DIR / "data" / "samples"
MULTI_SIGNAL_RAW = SAMPLES_DIR / "multi_signal_raw.csv"
CICIDS_FALLBACK = SAMPLES_DIR / "cicids_sample_stream.csv"

# 5 Orthogonal Signal Families from Review 2
FEATURE_GROUPS: Dict[str, List[str]] = {
    "Packet Volumes": [
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Fwd Packet Length Max",
        "Fwd Packet Length Min",
        "Fwd Packet Length Mean",
    ],
    "Flow Rates": [
        "Flow Bytes/s",
        "Flow Packets/s",
    ],
    "Timing & Inter-Arrival Time (IAT)": [
        "Flow Duration",
        "Flow IAT Mean",
        "Flow IAT Max",
        "Fwd IAT Mean",
    ],
    "TCP Flags & Headers": [
        "SYN Flag Count",
        "ACK Flag Count",
        "FIN Flag Count",
        "RST Flag Count",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
    ],
    "Active & Idle Behavior": [
        "Active Mean",
        "Idle Mean",
    ],
}

# Offline reported F1 benchmarks from Review 2
OFFLINE_BENCHMARKS = [
    {"name": "XGBoost Classifier", "type": "Supervised", "f1": 0.9970, "precision": 0.9975, "recall": 0.9965, "latency_ms": 1.2},
    {"name": "Random Forest", "type": "Supervised", "f1": 0.9947, "precision": 0.9950, "recall": 0.9944, "latency_ms": 3.8},
    {"name": "Decision Tree", "type": "Supervised", "f1": 0.9906, "precision": 0.9912, "recall": 0.9900, "latency_ms": 0.4},
    {"name": "Logistic Regression", "type": "Linear Supervised", "f1": 0.8334, "precision": 0.8410, "recall": 0.8260, "latency_ms": 0.2},
    {"name": "Standard Autoencoder", "type": "Unsupervised", "f1": 0.6937, "precision": 0.7020, "recall": 0.6856, "latency_ms": 1.9},
    {"name": "Multi-Signal Autoencoder (Fixed)", "type": "Unsupervised Multi-Signal", "f1": 0.6924, "precision": 0.7100, "recall": 0.6756, "latency_ms": 2.1},
    {"name": "Stabilized Adaptive Threshold", "type": "Streaming Self-Tuning", "f1": 0.6891, "precision": 0.7050, "recall": 0.6740, "latency_ms": 2.3},
    {"name": "Fixed Threshold Baseline", "type": "Static Anomaly", "f1": 0.6924, "precision": 0.7100, "recall": 0.6756, "latency_ms": 2.1},
    {"name": "Isolation Forest", "type": "Unsupervised", "f1": 0.5997, "precision": 0.6120, "recall": 0.5880, "latency_ms": 5.4},
    {"name": "Original Adaptive (Rolling MAD)", "type": "Streaming Unstable", "f1": 0.4079, "precision": 0.4210, "recall": 0.3955, "latency_ms": 2.2},
]


class DataStore:
    """Singleton store managing current dataset, detection results, and audit records."""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.dataset_name: str = "CICIDS2017 Working Hours Sample"
        self.detection_history: List[Dict[str, Any]] = []
        self._load_initial_dataset()

    def _load_initial_dataset(self) -> None:
        if MULTI_SIGNAL_RAW.exists():
            self.df = pd.read_csv(MULTI_SIGNAL_RAW)
            self.dataset_name = "CSE-CIC-IDS2018 / CICIDS2017 Sample (32.7k flows)"
        elif CICIDS_FALLBACK.exists():
            self.df = pd.read_csv(CICIDS_FALLBACK)
            self.dataset_name = "CICIDS2017 Benchmark Stream (1,500 flows)"
        else:
            # Synthetic fallback with realistic CICIDS2017 schema
            np.random.seed(42)
            n = 5000
            self.df = pd.DataFrame({
                "Flow Duration": np.random.exponential(50000, n),
                "Total Fwd Packets": np.random.poisson(6, n) + 1,
                "Total Backward Packets": np.random.poisson(5, n),
                "Total Length of Fwd Packets": np.random.uniform(100, 5000, n),
                "Flow Bytes/s": np.random.exponential(10000, n),
                "Flow Packets/s": np.random.exponential(200, n),
                "Destination Port": np.random.choice([80, 443, 22, 53, 8080], n),
                "Label": ["BENIGN"] * int(n * 0.75) + ["DoS Hulk"] * int(n * 0.15) + ["PortScan"] * (n - int(n * 0.75) - int(n * 0.15))
            })
            self.dataset_name = "CICIDS2017 Memory Sample"

    def set_dataset(self, df: pd.DataFrame, name: str) -> None:
        self.df = df
        self.dataset_name = name
        self.detection_history.clear()

    def get_summary(self) -> Dict[str, Any]:
        if self.df is None or self.df.empty:
            return {"rows": 0, "columns": 0, "null_count": 0, "inf_count": 0, "duplicates": 0, "name": "Empty"}

        total_rows = len(self.df)
        total_cols = self.df.shape[1]
        null_count = int(self.df.isna().sum().sum())
        num_df = self.df.select_dtypes(include=[np.number])
        inf_count = int(np.isinf(num_df.to_numpy()).sum()) if not num_df.empty else 0
        dup_count = int(self.df.duplicated().sum())

        class_dist = {}
        if "Label" in self.df.columns:
            class_dist = self.df["Label"].value_counts().to_dict()

        return {
            "name": self.dataset_name,
            "total_rows": total_rows,
            "total_columns": total_cols,
            "null_count": null_count,
            "inf_count": inf_count,
            "duplicate_count": dup_count,
            "class_distribution": class_dist,
            "numeric_feature_count": len(num_df.columns),
            "signal_families": list(FEATURE_GROUPS.keys())
        }

    def get_features(self) -> List[Dict[str, Any]]:
        if self.df is None:
            return []

        features = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            missing = int(self.df[col].isna().sum())
            is_num = pd.api.types.is_numeric_dtype(self.df[col])
            stats = {}
            if is_num:
                clean_vals = self.df[col].dropna()
                if not clean_vals.empty:
                    stats = {
                        "min": float(np.round(clean_vals.min(), 3)),
                        "max": float(np.round(clean_vals.max(), 3)),
                        "mean": float(np.round(clean_vals.mean(), 3)),
                        "std": float(np.round(clean_vals.std(), 3))
                    }
            features.append({
                "name": col,
                "type": dtype,
                "missing": missing,
                "stats": stats
            })
        return features


# Global singleton instance
data_store = DataStore()
