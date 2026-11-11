"""
Streaming Simulation and Ring-Buffer Engine.

Provides high-frequency packet generation, sliding buffer memory, and streaming state.
"""

from typing import List, Dict, Any
import numpy as np


class StreamingEngine:
    """Simulates real-time network flow ingestion using a ring buffer."""

    def __init__(self, max_buffer_size: int = 200) -> None:
        self.max_buffer_size: int = max_buffer_size
        self.timestamps: List[int] = []
        self.scores: List[float] = []
        self.reconstruction_errors: List[float] = []
        self.latent_deviations: List[float] = []
        self.thresholds_fixed: List[float] = []
        self.thresholds_original: List[float] = []
        self.thresholds_stabilized: List[float] = []
        self.labels: List[int] = []  # 0: Normal, 1: Attack

    def append_event(
        self,
        step: int,
        score: float,
        rec_err: float,
        lat_dev: float,
        th_fixed: float,
        th_orig: float,
        th_stab: float,
        label: int = 0,
    ) -> None:
        """Appends a new streaming telemetry step to the internal ring buffer."""
        self.timestamps.append(step)
        self.scores.append(score)
        self.reconstruction_errors.append(rec_err)
        self.latent_deviations.append(lat_dev)
        self.thresholds_fixed.append(th_fixed)
        self.thresholds_original.append(th_orig)
        self.thresholds_stabilized.append(th_stab)
        self.labels.append(label)

        # Enforce buffer capacity
        if len(self.timestamps) > self.max_buffer_size:
            self.timestamps.pop(0)
            self.scores.pop(0)
            self.reconstruction_errors.pop(0)
            self.latent_deviations.pop(0)
            self.thresholds_fixed.pop(0)
            self.thresholds_original.pop(0)
            self.thresholds_stabilized.pop(0)
            self.labels.pop(0)

    def to_dict(self) -> Dict[str, Any]:
        """Exports the buffer state as a dictionary for Plotly charts."""
        return {
            "step": self.timestamps,
            "score": self.scores,
            "rec_err": self.reconstruction_errors,
            "lat_dev": self.latent_deviations,
            "th_fixed": self.thresholds_fixed,
            "th_orig": self.thresholds_original,
            "th_stab": self.thresholds_stabilized,
            "label": self.labels,
        }

