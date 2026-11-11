"""
Deep Autoencoder Model Definition & Multi-Signal Inference Engine.
Reconstructs the PyTorch architecture from Review 2:
78 -> 128 -> 64 -> 16 -> 64 -> 128 -> 78
Combines Reconstruction Error (e_r) + Latent-Space Deviation (d_z).
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from joblib import load

# Check both locations for saved weights
BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "model_outputs"
WEIGHTS_DIR = BASE_DIR / "core" / "models" / "weights"


def get_checkpoint_path() -> Path:
    if (OUTPUT_DIR / "deep_autoencoder.pt").exists():
        return OUTPUT_DIR / "deep_autoencoder.pt"
    return WEIGHTS_DIR / "deep_autoencoder.pt"


def get_pipeline_path() -> Path:
    if (OUTPUT_DIR / "preprocessing_pipeline.joblib").exists():
        return OUTPUT_DIR / "preprocessing_pipeline.joblib"
    return WEIGHTS_DIR / "preprocessing_pipeline.joblib"


class DeepAutoencoder(nn.Module):
    """
    Exact PyTorch Deep Autoencoder architecture:
    Encoder: 78 -> 128 -> 64 -> 16 (bottleneck latent representation z)
    Decoder: 16 -> 64 -> 128 -> 78 (reconstruction x_hat)
    """

    def __init__(self, input_dim: int = 78):
        super().__init__()
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


class MultiSignalInferenceService:
    """
    Inference service for Deep Autoencoder.
    Calculates:
    1. Reconstruction Error (e_r)
    2. Latent-Space Deviation (d_z)
    3. Multi-Signal Composite Score (S_t)
    """

    def __init__(self):
        self.device = torch.device("cpu")
        self.model: Optional[DeepAutoencoder] = None
        self.pipeline: Optional[Dict[str, Any]] = None
        self.calibration: Optional[Dict[str, Any]] = None
        self._load_artifacts()

    def reload_artifacts(self) -> None:
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        pipe_path = get_pipeline_path()
        if pipe_path.exists():
            try:
                self.pipeline = load(pipe_path)
            except Exception as e:
                print(f"Warning loading pipeline: {e}")

        ckpt_path = get_checkpoint_path()
        if ckpt_path.exists():
            try:
                checkpoint = torch.load(
                    ckpt_path, map_location=self.device, weights_only=False
                )
                input_dim = checkpoint.get("input_dim", 78)
                self.model = DeepAutoencoder(input_dim).to(self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()

                self.calibration = {
                    "latent_mean": checkpoint.get("latent_mean", np.zeros(16)),
                    "latent_std": checkpoint.get("latent_std", np.ones(16)),
                    "median_r": checkpoint.get("median_r", 0.35),
                    "mad_r": checkpoint.get("mad_r", 0.12),
                    "median_l": checkpoint.get("median_l", 1.0),
                    "mad_l": checkpoint.get("mad_l", 0.4),
                    "base_threshold": checkpoint.get("base_threshold", 32.7688),
                    "best_k": checkpoint.get("best_k", 3.0),
                    "lower_ratio": checkpoint.get("lower_ratio", 0.8),
                    "upper_ratio": checkpoint.get("upper_ratio", 1.2),
                    "alpha_smooth": checkpoint.get("alpha_smooth", 0.05),
                }
            except Exception as e:
                print(f"Warning loading checkpoint: {e}")

        # Fallback initialization if artifacts not yet present
        if self.model is None:
            self.model = DeepAutoencoder(78).to(self.device)
            self.model.eval()
            self.calibration = {
                "latent_mean": np.zeros(16),
                "latent_std": np.ones(16),
                "median_r": 0.35,
                "mad_r": 0.12,
                "median_l": 1.0,
                "mad_l": 0.4,
                "base_threshold": 32.7688,
                "best_k": 3.0,
                "lower_ratio": 0.8,
                "upper_ratio": 1.2,
                "alpha_smooth": 0.05,
            }

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def predict_signals(
        self, X_raw: np.ndarray, alpha_weight: float = 0.50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Executes forward pass.
        Returns (reconstruction_errors, latent_deviations, combined_scores)
        """
        if not self.is_ready:
            raise RuntimeError("Model is not initialized.")

        # Preprocessing scaling with robust inf/nan guard
        X_raw = np.asarray(X_raw, dtype=np.float64)
        X_raw = np.where(np.isinf(X_raw), np.nan, X_raw)

        if self.pipeline is not None and "imputer" in self.pipeline and "scaler" in self.pipeline:
            imputer = self.pipeline["imputer"]
            if not hasattr(imputer, "_fill_dtype") and hasattr(imputer, "statistics_"):
                imputer._fill_dtype = imputer.statistics_.dtype
            
            try:
                X_imp = imputer.transform(X_raw)
            except Exception:
                if hasattr(imputer, "statistics_"):
                    stats = imputer.statistics_
                    if stats.shape[0] == X_raw.shape[1]:
                        X_imp = np.where(np.isnan(X_raw), stats, X_raw)
                    else:
                        X_imp = np.nan_to_num(X_raw, nan=0.0)
                else:
                    X_imp = np.nan_to_num(X_raw, nan=0.0)

            scaler = self.pipeline["scaler"]
            try:
                X_sc = scaler.transform(X_imp)
            except Exception:
                if hasattr(scaler, "mean_") and hasattr(scaler, "scale_") and scaler.mean_.shape[0] == X_imp.shape[1]:
                    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
                    X_sc = (X_imp - scaler.mean_) / scale
                else:
                    X_clean = np.nan_to_num(X_imp, nan=0.0)
                    mean = np.mean(X_clean, axis=0, keepdims=True)
                    std = np.std(X_clean, axis=0, keepdims=True) + 1e-6
                    X_sc = (X_clean - mean) / std

            X_sc = np.clip(X_sc, -10.0, 10.0).astype(np.float32)
        else:
            # Robust fallback z-score
            X_clean = np.nan_to_num(X_raw, nan=0.0, posinf=1e4, neginf=-1e4)
            mean = np.mean(X_clean, axis=0, keepdims=True)
            std = np.std(X_clean, axis=0, keepdims=True) + 1e-6
            X_sc = np.clip((X_clean - mean) / std, -10.0, 10.0).astype(np.float32)

        tensor_X = torch.tensor(X_sc, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            reconstructed, latent = self.model(tensor_X)
            # 1. Reconstruction error (MSE per sample)
            error = torch.mean((tensor_X - reconstructed) ** 2, dim=1).cpu().numpy()
            latent_np = latent.cpu().numpy()

        # 2. Latent space deviation
        latent_mean = self.calibration["latent_mean"]
        latent_std = self.calibration["latent_std"]
        norm_latent = (latent_np - latent_mean) / (latent_std + 1e-6)
        deviation = np.sqrt(np.mean(norm_latent ** 2, axis=1))

        # 3. Standardized Multi-Signal Score Fusion
        median_r, mad_r = self.calibration["median_r"], self.calibration["mad_r"]
        median_l, mad_l = self.calibration["median_l"], self.calibration["mad_l"]

        norm_r = (error - median_r) / max(mad_r * 1.4826, 1e-6)
        norm_l = (deviation - median_l) / max(mad_l * 1.4826, 1e-6)

        combined_score = alpha_weight * norm_r + (1.0 - alpha_weight) * norm_l
        return error, deviation, combined_score


# Singleton Service Instance
service = MultiSignalInferenceService()
