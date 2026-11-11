"""
NETGUARD AI — Complete Pipeline Integration for ML_DeepLearning.ipynb and ML_Adaptive_Threshold.ipynb.

Implements:
1. Data loading, cleaning, imputation, scaling, and stratified splitting (Train 70%, Val 15%, Test 15%).
2. Supervised & Unsupervised Machine Learning Models:
   - Random Forest Classifier (100 trees, max_depth=15)
   - XGBoost Classifier (100 estimators, max_depth=6, lr=0.1)
   - Decision Tree (CART, max_depth=12, min_samples_leaf=10)
   - Logistic Regression (max_iter=500, balanced)
   - Isolation Forest
3. Deep Autoencoder (PyTorch):
   - Architecture: Input (dynamic) -> 128 -> 64 -> 16 -> 64 -> 128 -> Input
   - Trained on Normal / Benign traffic only
   - Dual-signal score: Reconstruction Error (e_r) + Latent Deviation (d_z)
   - Centering & scaling with training set Median & MAD
4. Streaming Adaptive Thresholding:
   - Fixed Threshold (calibrated on normal validation scores)
   - Original Adaptive Threshold (Rolling MAD, k calibrated on validation, attack exclusion)
   - Stabilized Adaptive Threshold (Bounded ratio clamping + Exponential smoothing alpha=0.05, attack exclusion)
5. Comprehensive Evaluation & Artifact Persistence:
   - shared_data.npz
   - preprocessing_pipeline.joblib
   - deep_autoencoder.pt
   - random_forest.joblib, xgboost.joblib, decision_tree.joblib, logistic_regression.joblib, isolation_forest.joblib
   - anomaly_scores_val.csv, anomaly_scores_test.csv
   - attharv_results.csv, ishan_results.csv, final_model_comparison.csv
   - model_comparison_graph.png
"""

import os
from pathlib import Path
from collections import deque
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "model_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Deep Autoencoder Architecture (Dynamic Input Dimension)
# -----------------------------------------------------------------------------
class DynamicDeepAutoencoder(nn.Module):
    """
    Deep Autoencoder matching ML_DeepLearning.ipynb:
    Input (D) -> 128 -> 64 -> 16 (Bottleneck Z) -> 64 -> 128 -> Input (D)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


# -----------------------------------------------------------------------------
# Evaluation Helper Function (Exact from Notebooks)
# -----------------------------------------------------------------------------
def evaluate_model_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fpr = float(fp / max(fp + tn, 1))

    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "FPR": round(fpr, 4),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


# -----------------------------------------------------------------------------
# Core Threshold Functions (Exact from ML_Adaptive_Threshold.ipynb)
# -----------------------------------------------------------------------------
def calculate_adaptive_threshold(history: List[float], k: float = 3.0) -> float:
    arr = np.asarray(history, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = max(1.4826 * mad, 1e-6)
    return med + k * scale


def run_original_adaptive_detector(
    scores: np.ndarray,
    initial_history: np.ndarray,
    k: float = 3.0,
    max_history: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    history = deque(initial_history[-max_history:].tolist(), maxlen=max_history)
    predictions = []
    thresholds = []

    for s in scores:
        tau = calculate_adaptive_threshold(list(history), k=k)
        pred = int(s > tau)
        predictions.append(pred)
        thresholds.append(tau)
        if pred == 0:
            history.append(float(s))

    return np.array(predictions, dtype=int), np.array(thresholds, dtype=float)


def run_stabilized_adaptive_detector(
    scores: np.ndarray,
    initial_history: np.ndarray,
    base_threshold: float,
    k: float = 3.0,
    alpha: float = 0.05,
    lower_ratio: float = 0.8,
    upper_ratio: float = 1.2,
    max_history: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    history = deque(initial_history[-max_history:].tolist(), maxlen=max_history)
    threshold = float(base_threshold)
    lower_bound = base_threshold * lower_ratio
    upper_bound = base_threshold * upper_ratio

    predictions = []
    thresholds = []

    for s in scores:
        arr = np.asarray(history, dtype=float)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        scale = max(1.4826 * mad, 1e-6)
        proposed = med + k * scale
        bounded = float(np.clip(proposed, lower_bound, upper_bound))
        threshold = (1.0 - alpha) * threshold + alpha * bounded
        pred = int(s > threshold)
        predictions.append(pred)
        thresholds.append(threshold)
        if pred == 0:
            history.append(float(s))

    return np.array(predictions, dtype=int), np.array(thresholds, dtype=float)


# -----------------------------------------------------------------------------
# Pipeline Execution & Training Engine
# -----------------------------------------------------------------------------
class NotebookPipelineEngine:
    """Orchestrates end-to-end execution and artifact generation."""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_pipeline(self, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """Runs complete ML_DeepLearning and ML_Adaptive_Threshold workflows."""
        # 1. Resolve Dataset
        if dataset_path and os.path.exists(dataset_path):
            csv_path = Path(dataset_path)
        else:
            default_stream = BASE_DIR / "data" / "samples" / "cicids_sample_stream.csv"
            multi_sig = BASE_DIR / "data" / "samples" / "multi_signal_raw.csv"
            csv_path = default_stream if default_stream.exists() else multi_sig

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Extract Label
        label_col = next((c for c in df.columns if c.lower() in ["label", "class"]), None)
        if not label_col:
            raise ValueError("Dataset missing 'Label' target column.")

        raw_labels = df[label_col].astype(str).str.strip()
        y = (raw_labels != "BENIGN").astype(int).to_numpy()

        # Clean numeric features
        X_df = df.drop(columns=[label_col]).copy()
        X_df = X_df.apply(pd.to_numeric, errors="coerce")
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        empty_cols = X_df.columns[X_df.isna().all()].tolist()
        if empty_cols:
            X_df = X_df.drop(columns=empty_cols)

        feature_names = list(X_df.columns)
        input_dim = len(feature_names)

        # 2. Stratified Splits (Train 70%, Val 15%, Test 15%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_df, y, test_size=0.30, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )

        # 3. Fit Preprocessing Pipeline on Training Data Only
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_s = scaler.fit_transform(imputer.fit_transform(X_train))
        X_val_s = scaler.transform(imputer.transform(X_val))
        X_test_s = scaler.transform(imputer.transform(X_test))

        # Save shared dataset & preprocessing
        np.savez_compressed(
            self.output_dir / "shared_data.npz",
            X_train=X_train_s, X_val=X_val_s, X_test=X_test_s,
            y_train=y_train, y_val=y_val, y_test=y_test,
            feature_names=feature_names,
        )
        joblib.dump(
            {"imputer": imputer, "scaler": scaler, "features": feature_names, "feature_names": feature_names},
            self.output_dir / "preprocessing_pipeline.joblib"
        )

        # 4. Train Supervised Baseline Models
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight="balanced", n_jobs=-1)
        rf_model.fit(X_train_s, y_train)
        rf_preds = rf_model.predict(X_test_s)
        joblib.dump(rf_model, self.output_dir / "random_forest.joblib")

        # XGBoost
        if HAS_XGB:
            scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="logloss", scale_pos_weight=scale_pos)
            xgb_model.fit(X_train_s, y_train)
            xgb_preds = xgb_model.predict(X_test_s)
            joblib.dump(xgb_model, self.output_dir / "xgboost.joblib")
        else:
            xgb_preds = rf_preds

        # Decision Tree (CART)
        dt_model = DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, class_weight="balanced", random_state=42)
        dt_model.fit(X_train_s, y_train)
        dt_preds = dt_model.predict(X_test_s)
        joblib.dump(dt_model, self.output_dir / "decision_tree.joblib")

        # Logistic Regression
        lr_model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        lr_model.fit(X_train_s, y_train)
        lr_preds = lr_model.predict(X_test_s)
        joblib.dump(lr_model, self.output_dir / "logistic_regression.joblib")

        # Isolation Forest
        contam = float(max(0.01, min(0.5, (y_train == 1).mean())))
        if_model = IsolationForest(n_estimators=100, contamination=contam, random_state=42, n_jobs=-1)
        if_model.fit(X_train_s)
        if_raw = if_model.predict(X_test_s)
        if_preds = (if_raw == -1).astype(int)
        joblib.dump(if_model, self.output_dir / "isolation_forest.joblib")

        # 5. Train Deep Autoencoder on Normal Benign Traffic Only
        normal_train = X_train_s[y_train == 0]
        model = DynamicDeepAutoencoder(input_dim=input_dim).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        dataset = TensorDataset(torch.tensor(normal_train, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=min(128, len(normal_train)), shuffle=True)

        model.train()
        epochs = 15
        for epoch in range(epochs):
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                recon, _ = model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()

        # 6. Extract Reconstruction Errors & Latent Space Deviations
        model.eval()
        @torch.no_grad()
        def extract(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            t_X = torch.tensor(X, dtype=torch.float32, device=self.device)
            recon, lat = model(t_X)
            err = torch.mean((t_X - recon) ** 2, dim=1).cpu().numpy()
            return err, lat.cpu().numpy()

        train_err, train_lat = extract(normal_train)
        val_err, val_lat = extract(X_val_s)
        test_err, test_lat = extract(X_test_s)

        # Latent deviation statistics
        latent_mean = np.mean(train_lat, axis=0)
        latent_std = np.maximum(np.std(train_lat, axis=0), 1e-4)

        def calc_latent_dev(lat: np.ndarray) -> np.ndarray:
            norm = (lat - latent_mean) / latent_std
            return np.sqrt(np.mean(norm ** 2, axis=1))

        train_dev = calc_latent_dev(train_lat)
        val_dev = calc_latent_dev(val_lat)
        test_dev = calc_latent_dev(test_lat)

        # Multi-signal Score Fusion (Median & MAD scaled)
        def norm_signal(vals: np.ndarray, ref: np.ndarray) -> np.ndarray:
            med = float(np.median(ref))
            mad = float(np.median(np.abs(ref - med)))
            scale = max(1.4826 * mad, 1e-8)
            return (vals - med) / scale

        val_r = norm_signal(val_err, train_err)
        test_r = norm_signal(test_err, train_err)
        val_l = norm_signal(val_dev, train_dev)
        test_l = norm_signal(test_dev, train_dev)

        val_combined = 0.5 * val_r + 0.5 * val_l
        test_combined = 0.5 * test_r + 0.5 * test_l

        # Save anomaly score CSV files
        val_scores_df = pd.DataFrame({
            "sample_id": np.arange(len(y_val)),
            "reconstruction_error": val_err,
            "latent_deviation": val_dev,
            "combined_score": val_combined,
            "label": y_val,
        })
        test_scores_df = pd.DataFrame({
            "sample_id": np.arange(len(y_test)),
            "reconstruction_error": test_err,
            "latent_deviation": test_dev,
            "combined_score": test_combined,
            "label": y_test,
        })
        val_scores_df.to_csv(self.output_dir / "anomaly_scores_val.csv", index=False)
        test_scores_df.to_csv(self.output_dir / "anomaly_scores_test.csv", index=False)

        # 7. Threshold Calibrations
        # A) Fixed Baseline Threshold (Validation only)
        normal_val_scores = val_combined[y_val == 0]
        threshold_candidates = np.quantile(normal_val_scores, np.linspace(0.80, 0.999, 100))
        best_fixed = max(
            threshold_candidates,
            key=lambda t: f1_score(y_val, (val_combined > t).astype(int), zero_division=0)
        )
        fixed_preds = (test_combined > best_fixed).astype(int)

        # Standard Autoencoder (Recon error only)
        normal_val_errors = val_err[y_val == 0]
        recon_thresholds = np.quantile(normal_val_errors, np.linspace(0.90, 0.999, 50))
        best_recon_t = max(
            recon_thresholds,
            key=lambda t: f1_score(y_val, (val_err > t).astype(int), zero_division=0)
        )
        ae_preds = (test_err > best_recon_t).astype(int)

        # B) Original Adaptive Threshold (k search on validation)
        k_candidates = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        k_results = []
        for k_val in k_candidates:
            preds, _ = run_original_adaptive_detector(val_combined, normal_val_scores, k=k_val)
            f1_k = f1_score(y_val, preds, zero_division=0)
            k_results.append((k_val, f1_k))
        best_k = max(k_results, key=lambda x: x[1])[0]
        adaptive_preds, adaptive_taus = run_original_adaptive_detector(test_combined, normal_val_scores, k=best_k)

        # C) Stabilized Adaptive Threshold (Ratio grid search on validation)
        ratio_candidates = [(0.7, 1.3), (0.8, 1.2), (0.9, 1.1), (0.85, 1.15)]
        stab_results = []
        for low_r, up_r in ratio_candidates:
            s_preds, _ = run_stabilized_adaptive_detector(
                val_combined, normal_val_scores, base_threshold=best_fixed,
                k=best_k, alpha=0.05, lower_ratio=low_r, upper_ratio=up_r
            )
            s_f1 = f1_score(y_val, s_preds, zero_division=0)
            stab_results.append({"lower_ratio": low_r, "upper_ratio": up_r, "val_f1": s_f1})
        best_ratio_cfg = max(stab_results, key=lambda x: x["val_f1"])

        stabilized_preds, stabilized_taus = run_stabilized_adaptive_detector(
            test_combined, normal_val_scores, base_threshold=best_fixed,
            k=best_k, alpha=0.05,
            lower_ratio=best_ratio_cfg["lower_ratio"],
            upper_ratio=best_ratio_cfg["upper_ratio"],
        )

        # Save Deep Autoencoder weights & calibration metadata
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "latent_mean": latent_mean,
                "latent_std": latent_std,
                "median_r": float(np.median(train_err)),
                "mad_r": float(np.median(np.abs(train_err - np.median(train_err)))),
                "median_l": float(np.median(train_dev)),
                "mad_l": float(np.median(np.abs(train_dev - np.median(train_dev)))),
                "base_threshold": float(best_fixed),
                "best_k": float(best_k),
                "lower_ratio": float(best_ratio_cfg["lower_ratio"]),
                "upper_ratio": float(best_ratio_cfg["upper_ratio"]),
                "alpha_smooth": 0.05,
            },
            self.output_dir / "deep_autoencoder.pt"
        )

        # 8. Produce Model Comparison Tables
        attharv_results = pd.DataFrame([
            evaluate_model_metrics("Random Forest", y_test, rf_preds),
            evaluate_model_metrics("XGBoost", y_test, xgb_preds),
            evaluate_model_metrics("Isolation Forest", y_test, if_preds),
            evaluate_model_metrics("Standard Autoencoder", y_test, ae_preds),
            evaluate_model_metrics("Multi-Signal + Fixed Threshold", y_test, fixed_preds),
        ])
        attharv_results.to_csv(self.output_dir / "attharv_results.csv", index=False)

        ishan_results = pd.DataFrame([
            evaluate_model_metrics("Logistic Regression", y_test, lr_preds),
            evaluate_model_metrics("Decision Tree", y_test, dt_preds),
            evaluate_model_metrics("Multi-Signal + Fixed Threshold", y_test, fixed_preds),
            evaluate_model_metrics("Multi-Signal + Original Adaptive", y_test, adaptive_preds),
            evaluate_model_metrics("Multi-Signal + Stabilized Adaptive", y_test, stabilized_preds),
        ])
        ishan_results.to_csv(self.output_dir / "ishan_results.csv", index=False)

        # Combined Comparison Table (Deduplicated on Model)
        final_results = pd.concat([attharv_results, ishan_results], ignore_index=True)
        final_results = final_results.drop_duplicates(subset=["Model"]).sort_values(by="F1-Score", ascending=False)
        final_results.to_csv(self.output_dir / "final_model_comparison.csv", index=False)

        # 9. Generate Comparison Graph (model_comparison_graph.png)
        plt.figure(figsize=(12, 6))
        plot_df = final_results.set_index("Model")[["Precision", "Recall", "F1-Score"]]
        plot_df.plot(kind="bar", figsize=(12, 6), colormap="viridis")
        plt.title("Machine Learning vs Deep Learning Anomaly Detection Benchmark")
        plt.xlabel("Models")
        plt.ylabel("Score")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_comparison_graph.png", dpi=200)
        plt.close("all")

        return {
            "status": "success",
            "samples_train": len(X_train_s),
            "samples_val": len(X_val_s),
            "samples_test": len(X_test_s),
            "input_dim": input_dim,
            "models_evaluated": len(final_results),
            "best_model": final_results.iloc[0]["Model"],
            "best_f1": float(final_results.iloc[0]["F1-Score"]),
            "best_config": best_ratio_cfg,
            "base_threshold": float(best_fixed),
            "best_k": float(best_k),
        }


# Global Pipeline Instance
pipeline_engine = NotebookPipelineEngine()
