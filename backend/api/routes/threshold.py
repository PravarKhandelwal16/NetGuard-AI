"""
Threshold Analysis API Routes.
Simulates and evaluates streaming threshold trajectories across Fixed,
Original Adaptive (Rolling MAD), and Stabilized Adaptive (Bounded + Smoothed).
"""

from pathlib import Path
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
from backend.core.thresholds import (
    FixedThresholdEngine,
    OriginalAdaptiveEngine,
    StabilizedAdaptiveEngine,
)

router = APIRouter(prefix="/threshold", tags=["Threshold Analysis"])

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "model_outputs"
TEST_SCORES_CSV = OUTPUT_DIR / "anomaly_scores_test.csv"


class CompareRequest(BaseModel):
    window_size: int = Field(50, ge=10, le=500, description="Rolling window size W")
    k_factor: float = Field(3.0, ge=1.0, le=6.0, description="MAD multiplier k")
    base_tau: float = Field(32.7688, description="Base threshold tau_0")
    alpha_smooth: float = Field(0.05, ge=0.01, le=0.5, description="Exponential smoothing rate")
    inject_burst: bool = Field(True, description="Inject sustained adversarial burst to demonstrate runaway")
    stream_length: int = Field(120, ge=50, le=500, description="Number of streaming time steps")
    use_real_scores: bool = Field(False, description="Use actual multi-signal test scores from notebook pipeline")


@router.post("/compare")
async def compare_thresholds(req: CompareRequest):
    """
    Simulates streaming anomaly scores with optional attack burst.
    Computes step-by-step thresholds for all three engines to illustrate runaway vs stabilization.
    """
    if req.use_real_scores and TEST_SCORES_CSV.exists():
        df_real = pd.read_csv(TEST_SCORES_CSV)
        N = min(req.stream_length, len(df_real))
        scores = df_real["combined_score"].iloc[:N].to_numpy(dtype=float)
        labels = df_real["label"].iloc[:N].astype(int).tolist()
        burst_start = None
        burst_end = None
    else:
        np.random.seed(42)
        N = req.stream_length

        # Normal baseline scores: Gamma/lognormal distribution around 15-25
        scores = np.random.gamma(shape=9.0, scale=2.2, size=N)

        # Inject sustained attack burst between 35% and 65% of the stream
        burst_start = int(N * 0.35)
        burst_end = int(N * 0.65)
        labels = [0] * N

        if req.inject_burst:
            burst_magnitude = np.random.normal(loc=65.0, scale=8.0, size=(burst_end - burst_start))
            scores[burst_start:burst_end] = burst_magnitude
            for i in range(burst_start, burst_end):
                labels[i] = 1

        # Add isolated anomalies
        isolated_indices = [int(N * 0.15), int(N * 0.82), int(N * 0.90)]
        for idx in isolated_indices:
            if idx < N:
                scores[idx] = 72.0
                labels[idx] = 1

    # Instantiate the 3 engines
    fixed_engine = FixedThresholdEngine(base_threshold=req.base_tau)
    orig_engine = OriginalAdaptiveEngine(
        window_size=req.window_size,
        k=req.k_factor,
    )
    stab_engine = StabilizedAdaptiveEngine(
        base_threshold=req.base_tau,
        window_size=req.window_size,
        k=req.k_factor,
        alpha=req.alpha_smooth,
        lower_ratio=0.80,
        upper_ratio=1.20,
    )

    f_preds, fixed_taus = fixed_engine.evaluate_batch(scores)
    o_preds, orig_taus = orig_engine.evaluate_stream(scores)
    s_preds, stab_taus = stab_engine.evaluate_stream(scores)

    trajectory = []
    for t in range(N):
        s = float(scores[t])
        t_fixed = float(fixed_taus[t])
        t_orig = float(orig_taus[t])
        t_stab = float(stab_taus[t])

        trajectory.append({
            "step": t,
            "timestamp": f"{10 + (t // 60):02d}:{(t % 60):02d}:00",
            "anomaly_score": round(s, 2),
            "fixed_threshold": round(t_fixed, 2),
            "original_adaptive": round(t_orig, 2),
            "stabilized_adaptive": round(t_stab, 2),
            "is_burst": burst_start <= t < burst_end if req.inject_burst else False,
            "ground_truth": labels[t],
            "flagged_fixed": bool(f_preds[t]),
            "flagged_original": bool(o_preds[t]),
            "flagged_stabilized": bool(s_preds[t]),
        })

    # Variance and stability metrics
    orig_var = float(np.var(orig_taus))
    stab_var = float(np.var(stab_taus))
    fixed_var = 0.0

    # Benchmark metrics from Review 2
    benchmarks = {
        "fixed": {
            "name": "Fixed Baseline",
            "offline_f1": 0.6924,
            "threshold_variance": round(fixed_var, 4),
            "runaway_risk": "Low",
            "description": "Static threshold (tau = 32.7688). Zero variance but cannot adapt to organic concept drift.",
        },
        "original_adaptive": {
            "name": "Original Adaptive (Rolling MAD)",
            "offline_f1": 0.4079,
            "threshold_variance": round(orig_var, 4),
            "runaway_risk": "Critical (Runaway Confirmed)",
            "description": "Rolling Median + 3*MAD. In sustained attacks, threshold escalates rapidly, hiding intrusions.",
        },
        "stabilized_adaptive": {
            "name": "Stabilized Adaptive (Proposed)",
            "offline_f1": 0.6891,
            "threshold_variance": round(stab_var, 4),
            "runaway_risk": "Mitigated (Bounded Clamping)",
            "description": "Bounded within [0.8*tau_0, 1.2*tau_0] with exponential smoothing (alpha=0.05).",
        },
    }

    return {
        "trajectory": trajectory,
        "metrics": benchmarks,
        "simulation_params": {
            "window_size": req.window_size,
            "k_factor": req.k_factor,
            "alpha_smooth": req.alpha_smooth,
            "inject_burst": req.inject_burst,
            "burst_window": [burst_start, burst_end] if req.inject_burst else None,
        },
    }
