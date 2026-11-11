"""
NETGUARD AI — Streaming Thresholding Engines.
Implements:
1. Fixed Threshold Baseline (Static Reference)
2. Original Adaptive Threshold (Rolling Median + MAD with attack exclusion)
3. Stabilized Adaptive Threshold (Bounded ratio envelope + Exponential Inertia Smoothing)
"""

from collections import deque
from typing import Tuple, List, Dict, Any
import numpy as np


class FixedThresholdEngine:
    """Fixed reference threshold calibrated offline."""

    def __init__(self, base_threshold: float = 32.7688):
        self.base_threshold = float(base_threshold)

    def evaluate_batch(self, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(scores, dtype=float)
        predictions = (scores > self.base_threshold).astype(int)
        thresholds = np.full_like(scores, fill_value=self.base_threshold)
        return predictions, thresholds


class OriginalAdaptiveEngine:
    """
    Original Adaptive Threshold using rolling median + MAD:
    Threshold_t = Median(history) + k * 1.4826 * MAD(history)
    Susceptible to threshold runaway during prolonged volumetric bursts.
    """

    def __init__(self, window_size: int = 100, k: float = 3.0):
        self.window_size = int(window_size)
        self.k = float(k)

    def evaluate_stream(
        self, scores: np.ndarray, initial_scores: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(scores, dtype=float)
        if initial_scores is None or len(initial_scores) == 0:
            warm_up = scores[: min(50, len(scores))]
        else:
            warm_up = np.asarray(initial_scores, dtype=float)

        history = deque(warm_up[-self.window_size :].tolist(), maxlen=self.window_size)
        predictions: List[int] = []
        thresholds: List[float] = []

        for s in scores:
            arr = np.asarray(history, dtype=float)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            tau = med + self.k * 1.4826 * max(mad, 1e-6)
            pred = int(s > tau)

            predictions.append(pred)
            thresholds.append(tau)

            # Exclude flagged attacks to mitigate contamination
            if pred == 0:
                history.append(float(s))

        return np.array(predictions, dtype=int), np.array(thresholds, dtype=float)


class StabilizedAdaptiveEngine:
    """
    Stabilized Adaptive Threshold:
    - Bounded between base_threshold * lower_ratio and base_threshold * upper_ratio
    - Exponential inertia smoothing: tau_{t+1} = (1 - alpha) * tau_t + alpha * proposed
    - Excludes flagged attacks from rolling history deque
    """

    def __init__(
        self,
        base_threshold: float = 32.7688,
        k: float = 3.0,
        window_size: int = 100,
        alpha: float = 0.05,
        lower_ratio: float = 0.80,
        upper_ratio: float = 1.20,
    ):
        self.base_threshold = float(base_threshold)
        self.k = float(k)
        self.window_size = int(window_size)
        self.alpha = float(alpha)
        self.lower_ratio = float(lower_ratio)
        self.upper_ratio = float(upper_ratio)

    def evaluate_stream(
        self, scores: np.ndarray, initial_scores: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(scores, dtype=float)
        if initial_scores is None or len(initial_scores) == 0:
            warm_up = scores[: min(50, len(scores))]
        else:
            warm_up = np.asarray(initial_scores, dtype=float)

        history = deque(warm_up[-self.window_size :].tolist(), maxlen=self.window_size)
        threshold = float(self.base_threshold)
        lower_bound = self.base_threshold * self.lower_ratio
        upper_bound = self.base_threshold * self.upper_ratio

        predictions: List[int] = []
        thresholds: List[float] = []

        for s in scores:
            arr = np.asarray(history, dtype=float)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            raw_prop = med + self.k * 1.4826 * max(mad, 1e-6)

            # 1. Hard bound clipping
            bounded_prop = float(np.clip(raw_prop, lower_bound, upper_bound))

            # 2. Exponential inertia smoothing
            threshold = (1.0 - self.alpha) * threshold + self.alpha * bounded_prop
            pred = int(s > threshold)

            predictions.append(pred)
            thresholds.append(threshold)

            # Exclude flagged observations
            if pred == 0:
                history.append(float(s))

        return np.array(predictions, dtype=int), np.array(thresholds, dtype=float)
