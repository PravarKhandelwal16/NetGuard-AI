"""
Unit tests for Thresholding Algorithms.
Uses the actual Engine classes from core.thresholding.algorithms.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
from core.thresholding.algorithms import (
    FixedThresholdEngine,
    OriginalAdaptiveEngine,
    StabilizedAdaptiveEngine,
)


def test_fixed_threshold_above():
    """Score above threshold should be flagged as anomaly."""
    eng = FixedThresholdEngine(base_threshold=0.5)
    preds, thresholds = eng.evaluate_batch(np.array([0.6, 0.3, 0.9]))
    assert preds[0] == 1, "0.6 > 0.5 must be flagged"
    assert preds[1] == 0, "0.3 < 0.5 must be normal"
    assert preds[2] == 1, "0.9 > 0.5 must be flagged"
    assert all(t == 0.5 for t in thresholds), "Fixed threshold must be constant"


def test_fixed_threshold_at_boundary():
    """Score exactly at threshold should NOT be flagged (strict >)."""
    eng = FixedThresholdEngine(base_threshold=0.5)
    preds, _ = eng.evaluate_batch(np.array([0.5]))
    assert preds[0] == 0, "Score equal to threshold is not anomalous"


def test_original_adaptive_requires_initial_scores():
    """OriginalAdaptiveEngine must have a non-empty initial_scores."""
    eng = OriginalAdaptiveEngine(window_size=50, k=3.0)
    warm_up = np.array([0.3, 0.32, 0.28, 0.31])
    preds, thresholds = eng.evaluate_stream(np.array([0.6, 0.3]), initial_scores=warm_up)
    assert len(preds) == 2
    assert len(thresholds) == 2


def test_stabilized_adaptive_bounded():
    """Stabilized threshold must remain within [lower, upper] bounds."""
    base = 1.0
    lower_ratio, upper_ratio = 0.80, 1.20
    eng = StabilizedAdaptiveEngine(
        base_threshold=base,
        k=3.0,
        window_size=20,
        alpha=0.10,
        lower_ratio=lower_ratio,
        upper_ratio=upper_ratio,
    )
    warm_up = np.full(20, 0.5)
    # Inject extreme spikes — threshold should never exceed bounds
    scores = np.full(50, 50.0)
    _, thresholds = eng.evaluate_stream(scores, initial_scores=warm_up)
    lower_bound = base * lower_ratio
    upper_bound = base * upper_ratio
    assert all(lower_bound <= t <= upper_bound for t in thresholds), (
        f"Threshold out of [{lower_bound}, {upper_bound}]: {min(thresholds):.3f}–{max(thresholds):.3f}"
    )


def test_stabilized_does_not_flag_all_normal():
    """Under purely normal traffic the stabilized engine should flag very few flows."""
    eng = StabilizedAdaptiveEngine(base_threshold=0.65, k=3.0, window_size=50)
    normal_scores = np.random.default_rng(42).uniform(0.2, 0.4, 100)
    warm_up = normal_scores[:20]
    preds, _ = eng.evaluate_stream(normal_scores[20:], initial_scores=warm_up)
    # Expect fewer than 10% false positives on clean traffic
    fp_rate = preds.mean()
    assert fp_rate < 0.10, f"Too many false positives under normal traffic: {fp_rate:.2%}"


if __name__ == "__main__":
    test_fixed_threshold_above()
    test_fixed_threshold_at_boundary()
    test_original_adaptive_requires_initial_scores()
    test_stabilized_adaptive_bounded()
    test_stabilized_does_not_flag_all_normal()
    print("All threshold tests passed.")
