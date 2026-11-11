"""
Deep Autoencoder Model Definition & Inference Engine.
Reconstructs the PyTorch architecture from Attharv's ML_DeepLearning.ipynb.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from joblib import load

from config.settings import WEIGHTS_DIR

MODEL_CHECKPOINT_PATH = WEIGHTS_DIR / "deep_autoencoder.pt"
PIPELINE_PATH = WEIGHTS_DIR / "preprocessing_pipeline.joblib"


class DeepAutoencoder(nn.Module):
    """
    Exact PyTorch Deep Autoencoder from ML_DeepLearning.ipynb:
    Encoder: input_dim -> 128 -> 64 -> 16
    Decoder: 16 -> 64 -> 128 -> input_dim
    """

    def __init__(self, input_dim: int):
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
    Loads saved DeepAutoencoder weights and preprocessing pipelines to run
    reconstruction error and latent space deviation scoring.
    """

    def __init__(self):
        self.device = torch.device("cpu")
        self.model: Optional[DeepAutoencoder] = None
        self.pipeline: Optional[Dict[str, Any]] = None
        self.calibration: Optional[Dict[str, Any]] = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        if PIPELINE_PATH.exists():
            self.pipeline = load(PIPELINE_PATH)

        if MODEL_CHECKPOINT_PATH.exists():
            # weights_only=False required for checkpoints saved with numpy arrays embedded
            checkpoint = torch.load(
                MODEL_CHECKPOINT_PATH, map_location=self.device, weights_only=False
            )
            input_dim = checkpoint["input_dim"]
            self.model = DeepAutoencoder(input_dim).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            self.calibration = {
                "latent_mean": checkpoint["latent_mean"],
                "latent_std": checkpoint["latent_std"],
                "median_r": checkpoint["median_r"],
                "mad_r": checkpoint["mad_r"],
                "median_l": checkpoint["median_l"],
                "mad_l": checkpoint["mad_l"],
                "base_threshold": checkpoint.get("base_threshold", 32.7688),
                "best_k": checkpoint.get("best_k", 3.0),
                "lower_ratio": checkpoint.get("lower_ratio", 0.8),
                "upper_ratio": checkpoint.get("upper_ratio", 1.2),
                "alpha_smooth": checkpoint.get("alpha_smooth", 0.05),
            }

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.pipeline is not None

    def predict_signals(
        self, X_raw: np.ndarray, alpha_weight: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates:
        1. Reconstruction Error (e_r)
        2. Latent Space Deviation (d_z)
        3. Normalized Composite Score (S_t)
        """
        if not self.is_ready:
            raise RuntimeError("Model checkpoint or pipeline not loaded.")

        # Transform using saved imputer & scaler
        imputer = self.pipeline["imputer"]
        scaler = self.pipeline["scaler"]

        X_imp = imputer.transform(X_raw)
        X_sc = scaler.transform(X_imp)
        X_sc = np.clip(X_sc, -10, 10).astype(np.float32)

        tensor_X = torch.tensor(X_sc, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            reconstructed, latent = self.model(tensor_X)

            # 1. Reconstruction error (MSE per sample)
            error = torch.mean((tensor_X - reconstructed) ** 2, dim=1).numpy()
            latent_np = latent.numpy()

        # 2. Latent space deviation
        latent_mean = self.calibration["latent_mean"]
        latent_std = self.calibration["latent_std"]
        normalized_latent = (latent_np - latent_mean) / latent_std
        deviation = np.sqrt(np.mean(normalized_latent ** 2, axis=1))

        # 3. Normalized multi-signal score fusion
        # (values - median) / (1.4826 * mad)
        median_r, mad_r = self.calibration["median_r"], self.calibration["mad_r"]
        median_l, mad_l = self.calibration["median_l"], self.calibration["mad_l"]

        norm_r = (error - median_r) / mad_r
        norm_l = (deviation - median_l) / mad_l

        # Weighted composite score (from ML_DeepLearning.ipynb cell 21)
        combined_score = alpha_weight * norm_r + (1.0 - alpha_weight) * norm_l
        return error, deviation, combined_score
