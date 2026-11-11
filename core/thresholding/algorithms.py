"""
Actual Streaming Adaptive Threshold Algorithms.
Extracted directly from Ishan's ML_Adaptive_Threshold.ipynb.
"""

from collections import deque
from typing import Tuple, List, Union
import numpy as np


class FixedThresholdEngine:
    """Fixed baseline threshold evaluated on validation calibration."""

    def __init__(self, base_threshold: float = 32.7688):
        self.base_threshold = float(base_threshold)

    def evaluate_batch(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions = (scores > self.base_threshold).astype(int)
        thresholds = np.full_like(scores, fill_value=self.base_threshold)
        return predictions, thresholds


class OriginalAdaptiveEngine:
    """
    Original Adaptive Threshold using rolling median + MAD from cell 10:
    Threshold_t = Median(history) + k * 1.4826 * MAD(history)
    Excludes suspected attacks from future threshold updates.
    """

    def __init__(self, window_size: int = 500, k: float = 3.0):
        self.window_size = window_size
        self.k = float(k)

    def evaluate_stream(
        self, scores: np.ndarray, initial_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        history = deque(
            np.asarray(initial_scores, dtype=float)[-self.window_size :].tolist(),
            maxlen=self.window_size,
        )

        if not history:
            raise ValueError("Initial scores cannot be empty")

        predictions: List[int] = []
        thresholds: List[float] = []

        for score in scores:
            values = np.asarray(history, dtype=float)
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            threshold = median + self.k * 1.4826 * max(mad, 1e-8)
            prediction = int(score > threshold)

            predictions.append(prediction)
            thresholds.append(threshold)

            # Exclude suspected attacks from future threshold updates
            if prediction == 0:
                history.append(float(score))

        return np.array(predictions), np.array(thresholds)


class StabilizedAdaptiveEngine:
    """
    Stabilized Adaptive Threshold from cell 12:
    - Bounded between base_threshold * lower_ratio and base_threshold * upper_ratio
    - Inertia smoothing with alpha parameter
    - Excludes flagged attacks from history
    """

    def __init__(
        self,
        base_threshold: float = 32.7688,
        k: float = 3.0,
        window_size: int = 500,
        alpha: float = 0.05,
        lower_ratio: float = 0.80,
        upper_ratio: float = 1.20,
    ):
        self.base_threshold = float(base_threshold)
        self.k = float(k)
        self.window_size = window_size
        self.alpha = float(alpha)
        self.lower_ratio = float(lower_ratio)
        self.upper_ratio = float(upper_ratio)

    def evaluate_stream(
        self, scores: np.ndarray, initial_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        history = deque(
            np.asarray(initial_scores, dtype=float)[-self.window_size :],
            maxlen=self.window_size,
        )

        if len(history) == 0:
            raise ValueError("Initial scores cannot be empty")

        threshold = float(self.base_threshold)
        predictions: List[int] = []
        thresholds: List[float] = []

        lower_bound = self.base_threshold * self.lower_ratio
        upper_bound = self.base_threshold * self.upper_ratio

        for score in scores:
            values = np.asarray(history)
            median = np.median(values)
            mad = np.median(np.abs(values - median))

            proposed = median + self.k * 1.4826 * max(mad, 1e-8)
            # Clip proposed update to prevent runaway
            proposed = np.clip(proposed, lower_bound, upper_bound)

            # Exponential smoothing update
            threshold = (1.0 - self.alpha) * threshold + self.alpha * proposed
            prediction = int(score > threshold)

            predictions.append(prediction)
            thresholds.append(threshold)

            # Exclude observations flagged as attacks
            if prediction == 0:
                history.append(float(score))

        return np.array(predictions), np.array(thresholds)
