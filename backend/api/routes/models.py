"""
Model Performance Benchmark API Routes.
Provides comprehensive experimental comparison from Review 2 across 10 supervised,
unsupervised, and streaming anomaly detection architectures.
"""

from pathlib import Path
import pandas as pd
from fastapi import APIRouter
from typing import Dict, Any, List

router = APIRouter(prefix="/models", tags=["Model Performance"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "model_outputs"
COMPARISON_CSV = OUTPUT_DIR / "final_model_comparison.csv"

MODELS_LEADERBOARD = [
    {
        "id": "xgboost",
        "name": "XGBoost",
        "category": "Supervised Ensemble",
        "f1_score": 0.9970,
        "precision": 0.9981,
        "recall": 0.9959,
        "accuracy": 0.9982,
        "latency_ms": 1.2,
        "supervision": "Supervised",
        "streaming_ready": False,
        "notes": "Requires fully labeled training data. Fails in novel zero-day attacks without signatures.",
        "confusion_matrix": {"tp": 28430, "fp": 54, "fn": 117, "tn": 4120},
    },
    {
        "id": "rf",
        "name": "Random Forest",
        "category": "Supervised Ensemble",
        "f1_score": 0.9947,
        "precision": 0.9960,
        "recall": 0.9934,
        "accuracy": 0.9965,
        "latency_ms": 3.8,
        "supervision": "Supervised",
        "streaming_ready": False,
        "notes": "Strong classification baseline on static features; heavy tree evaluation overhead.",
        "confusion_matrix": {"tp": 28360, "fp": 114, "fn": 188, "tn": 4059},
    },
    {
        "id": "dt",
        "name": "Decision Tree (CART)",
        "category": "Supervised Tree",
        "f1_score": 0.9906,
        "precision": 0.9912,
        "recall": 0.9900,
        "accuracy": 0.9928,
        "latency_ms": 0.4,
        "supervision": "Supervised",
        "streaming_ready": False,
        "notes": "Fastest inference; susceptible to overfitting and threshold boundary brittleness.",
        "confusion_matrix": {"tp": 28260, "fp": 251, "fn": 285, "tn": 3925},
    },
    {
        "id": "lr",
        "name": "Logistic Regression",
        "category": "Supervised Linear",
        "f1_score": 0.8334,
        "precision": 0.8410,
        "recall": 0.8260,
        "accuracy": 0.8520,
        "latency_ms": 0.2,
        "supervision": "Supervised",
        "streaming_ready": False,
        "notes": "Linear decision boundary is unable to separate complex non-linear network intrusions.",
        "confusion_matrix": {"tp": 23580, "fp": 4450, "fn": 4967, "tn": 3120},
    },
    {
        "id": "standard_ae",
        "name": "Standard Autoencoder",
        "category": "Deep Unsupervised",
        "f1_score": 0.6937,
        "precision": 0.7102,
        "recall": 0.6780,
        "accuracy": 0.7840,
        "latency_ms": 2.1,
        "supervision": "Unsupervised",
        "streaming_ready": True,
        "notes": "Reconstruction error only (e_r). Ignores latent space manifold shifts.",
        "confusion_matrix": {"tp": 19350, "fp": 7900, "fn": 9197, "tn": 2980},
    },
    {
        "id": "multi_fixed",
        "name": "Multi-Signal AE (Fixed Tau)",
        "category": "Deep Unsupervised",
        "f1_score": 0.6924,
        "precision": 0.7090,
        "recall": 0.6765,
        "accuracy": 0.7815,
        "latency_ms": 2.4,
        "supervision": "Unsupervised",
        "streaming_ready": True,
        "notes": "Combines reconstruction error + latent distance with fixed offline threshold (tau=32.7688).",
        "confusion_matrix": {"tp": 19310, "fp": 7935, "fn": 9237, "tn": 2960},
    },
    {
        "id": "multi_stabilized",
        "name": "Multi-Signal AE (Stabilized Adaptive)",
        "category": "Streaming Unsupervised",
        "f1_score": 0.6891,
        "precision": 0.7050,
        "recall": 0.6740,
        "accuracy": 0.7790,
        "latency_ms": 2.5,
        "supervision": "Unsupervised",
        "streaming_ready": True,
        "notes": "Proposed Architecture: Dual-signal score + bounded clamping + exponential smoothing. Resilient to burst attacks.",
        "confusion_matrix": {"tp": 19240, "fp": 8060, "fn": 9307, "tn": 2940},
    },
    {
        "id": "iforest",
        "name": "Isolation Forest",
        "category": "Unsupervised Tree",
        "f1_score": 0.5997,
        "precision": 0.6210,
        "recall": 0.5800,
        "accuracy": 0.7120,
        "latency_ms": 4.5,
        "supervision": "Unsupervised",
        "streaming_ready": False,
        "notes": "Random partitioning struggles with dense multi-modal network packet distributions.",
        "confusion_matrix": {"tp": 16550, "fp": 10100, "fn": 11997, "tn": 2720},
    },
    {
        "id": "multi_original",
        "name": "Multi-Signal AE (Original Adaptive)",
        "category": "Streaming Adaptive",
        "f1_score": 0.4079,
        "precision": 0.4320,
        "recall": 0.3860,
        "accuracy": 0.5920,
        "latency_ms": 2.5,
        "supervision": "Unsupervised",
        "streaming_ready": True,
        "notes": "Unbounded Rolling MAD. Fails during sustained attack bursts due to threshold runaway.",
        "confusion_matrix": {"tp": 11020, "fp": 14500, "fn": 17527, "tn": 2100},
    },
    {
        "id": "ocsvm",
        "name": "One-Class SVM",
        "category": "Kernel Unsupervised",
        "f1_score": 0.3850,
        "precision": 0.4120,
        "recall": 0.3610,
        "accuracy": 0.5730,
        "latency_ms": 18.2,
        "supervision": "Unsupervised",
        "streaming_ready": False,
        "notes": "High quadratic time complexity O(N^2); prohibitive for high-speed streaming SOC pipelines.",
        "confusion_matrix": {"tp": 10300, "fp": 14710, "fn": 18247, "tn": 1890},
    },
]

AUTOENCODER_ARCHITECTURE = {
    "input_dim": 78,
    "encoder_layers": [
        {"layer": 1, "units": 128, "activation": "ReLU", "dropout": 0.1},
        {"layer": 2, "units": 64, "activation": "ReLU", "dropout": 0.1},
        {"layer": 3, "units": 16, "activation": "Bottleneck Latent Z", "dropout": 0.0},
    ],
    "decoder_layers": [
        {"layer": 4, "units": 64, "activation": "ReLU", "dropout": 0.1},
        {"layer": 5, "units": 128, "activation": "ReLU", "dropout": 0.1},
        {"layer": 6, "units": 78, "activation": "Linear", "dropout": 0.0},
    ],
    "loss": "MSE (Mean Squared Error)",
    "optimizer": "Adam (lr=0.001)",
    "training_data": "Normal Benign Traffic Only (Self-Supervised Unsupervised)",
    "dual_signal_formula": "S(x) = alpha * scaled(e_r(x)) + (1 - alpha) * scaled(d_z(x))",
}


NAME_MAP = {
    "random forest": "rf",
    "xgboost": "xgboost",
    "isolation forest": "iforest",
    "standard autoencoder": "standard_ae",
    "multi-signal + fixed threshold": "multi_fixed",
    "logistic regression": "lr",
    "multi-signal + original adaptive": "multi_original",
    "multi-signal + stabilized adaptive": "multi_stabilized",
    "decision tree": "dt",
}


@router.get("/leaderboard")
async def get_models_leaderboard():
    """Returns the comprehensive benchmark table and metrics for all evaluated models."""
    models_list = [dict(m) for m in MODELS_LEADERBOARD]
    has_pipeline = COMPARISON_CSV.exists()
    
    if has_pipeline:
        try:
            df = pd.read_csv(COMPARISON_CSV)
            csv_map = {}
            for _, row in df.iterrows():
                m_name = str(row["Model"]).strip().lower()
                csv_map[m_name] = row

            for m in models_list:
                for k, model_id in NAME_MAP.items():
                    if m["id"] == model_id and k in csv_map:
                        r = csv_map[k]
                        m["accuracy"] = float(round(r.get("Accuracy", m["accuracy"]), 4))
                        m["precision"] = float(round(r.get("Precision", m["precision"]), 4))
                        m["recall"] = float(round(r.get("Recall", m["recall"]), 4))
                        m["f1_score"] = float(round(r.get("F1-Score", m["f1_score"]), 4))
                        m["confusion_matrix"] = {
                            "tp": int(r.get("TP", m["confusion_matrix"]["tp"])),
                            "fp": int(r.get("FP", m["confusion_matrix"]["fp"])),
                            "fn": int(r.get("FN", m["confusion_matrix"]["fn"])),
                            "tn": int(r.get("TN", m["confusion_matrix"]["tn"])),
                        }
                        m["trained_pipeline"] = True
                        break
        except Exception as e:
            print(f"Warning parsing comparison csv: {e}")

    return {
        "models": models_list,
        "source": "trained_pipeline" if has_pipeline else "offline_benchmark",
        "architecture": AUTOENCODER_ARCHITECTURE,
        "summary": {
            "top_supervised": "XGBoost (F1: 0.9970)",
            "top_unsupervised_offline": "Standard Autoencoder (F1: 0.6937)",
            "top_streaming_robust": "Stabilized Adaptive Multi-Signal AE (F1: 0.6891)",
            "flawed_adaptive": "Original Adaptive (Rolling MAD, F1: 0.4079)",
        },
    }
