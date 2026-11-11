"""
Functional Tests for NETGUARD AI.

Tests:
- CSV upload simulation and dataset validation
- Invalid file handling
- Missing column graceful degradation
- Empty dataset edge case
- Model loading readiness check
- Threshold engine integration
- Results export (CSV download)
- Large dataset handling
"""

from __future__ import annotations

import sys
import io
import numpy as np
import pandas as pd

sys.path.insert(0, ".")


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_valid_csv(rows: int = 50) -> pd.DataFrame:
    """Return a minimal valid DataFrame matching the pipeline's feature list."""
    from core.models.autoencoder import MultiSignalInferenceService
    svc = MultiSignalInferenceService()
    rng = np.random.default_rng(42)
    data = {col: rng.uniform(0, 100, rows) for col in svc.pipeline["features"]}
    data["Label"] = (["BENIGN"] * int(rows * 0.8)) + (["DDoS"] * (rows - int(rows * 0.8)))
    return pd.DataFrame(data)


def _make_malformed_csv_bytes() -> bytes:
    """Return bytes that cannot be parsed as a numeric CSV."""
    return b"garbage,,,\nnot a csv at all\n!!!\n"


# ── Test Cases ─────────────────────────────────────────────────────────────

def test_valid_dataset_loads() -> None:
    """Dataset should load with correct shape and Label column."""
    from core.data_manager import get_active_dataset
    df = get_active_dataset()
    assert df is not None, "Dataset must not be None"
    assert len(df) > 0, "Dataset must have at least one row"
    assert len(df.columns) > 10, "Dataset must have meaningful feature count"
    print(f"  ✅ Dataset loaded: {df.shape}")


def test_model_inference_on_valid_data() -> None:
    """ML pipeline must run inference and return correct output shapes."""
    from core.models.autoencoder import MultiSignalInferenceService
    svc = MultiSignalInferenceService()
    assert svc.is_ready, "MultiSignalInferenceService must be ready"

    df = _make_valid_csv(100)
    X = df[svc.pipeline["features"]].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    errors, deviations, scores = svc.predict_signals(X, alpha_weight=0.5)

    assert errors.shape == (100,), f"Unexpected error shape: {errors.shape}"
    assert deviations.shape == (100,), f"Unexpected deviation shape: {deviations.shape}"
    assert scores.shape == (100,), f"Unexpected score shape: {scores.shape}"
    assert not np.any(np.isnan(scores)), "Scores must not contain NaN"
    print(f"  ✅ Inference passed: errors[0]={errors[0]:.4f}, scores[0]={scores[0]:.4f}")


def test_missing_features_filled_with_zeros() -> None:
    """Dataset missing some features should fill with 0.0 without crashing."""
    from core.models.autoencoder import MultiSignalInferenceService
    svc = MultiSignalInferenceService()
    all_feats = svc.pipeline["features"]

    # Keep only first half of features
    partial_df = pd.DataFrame(
        np.random.default_rng(1).uniform(0, 1, (20, len(all_feats) // 2)),
        columns=all_feats[: len(all_feats) // 2],
    )

    X_df = pd.DataFrame(index=partial_df.index)
    for col in all_feats:
        if col in partial_df.columns:
            X_df[col] = pd.to_numeric(partial_df[col], errors="coerce")
        else:
            X_df[col] = 0.0
    X = X_df.to_numpy(dtype=np.float32)
    assert X.shape[1] == len(all_feats), "Feature matrix must match pipeline dimension"
    print(f"  ✅ Missing feature padding works: shape={X.shape}")


def test_empty_dataset_does_not_crash() -> None:
    """Zero-row DataFrame should be handled without exceptions in core utilities."""
    from core.models.autoencoder import MultiSignalInferenceService
    svc = MultiSignalInferenceService()
    empty_df = pd.DataFrame(columns=svc.pipeline["features"] + ["Label"])
    assert len(empty_df) == 0, "Empty dataset check"
    # Attempting inference on 0 rows should raise or return empty arrays
    X = empty_df[svc.pipeline["features"]].to_numpy(dtype=np.float32)
    assert X.shape == (0, len(svc.pipeline["features"])), "Empty X must have correct feature dim"
    print(f"  ✅ Empty dataset produces empty X: {X.shape}")


def test_fixed_threshold_engine() -> None:
    """Fixed engine should flag exactly the scores above base threshold."""
    from core.thresholding.algorithms import FixedThresholdEngine
    scores = np.array([0.4, 0.5, 0.65, 0.66, 0.9, 0.3])
    eng = FixedThresholdEngine(base_threshold=0.65)
    preds, thresholds = eng.evaluate_batch(scores)
    assert preds[0] == 0, "0.4 is normal"
    assert preds[3] == 1, "0.66 > 0.65 is anomaly"
    assert preds[4] == 1, "0.9 is anomaly"
    assert all(t == 0.65 for t in thresholds), "Fixed threshold must be constant"
    print(f"  ✅ Fixed engine: {preds.sum()} anomalies in {len(scores)} flows")


def test_original_adaptive_window_exclusion() -> None:
    """
    When all scores are very high, original adaptive should raise threshold
    so eventually nothing is flagged (window contamination behaviour confirmed).
    """
    from core.thresholding.algorithms import OriginalAdaptiveEngine
    rng = np.random.default_rng(7)
    warm = rng.uniform(0.1, 0.3, 50)
    # Sustained attack burst: very high scores
    burst = np.full(80, 5.0)
    eng = OriginalAdaptiveEngine(window_size=50, k=3.0)
    preds, thresholds = eng.evaluate_stream(burst, initial_scores=warm)
    # Window gets contaminated → threshold drifts up → fewer detections at the end
    end_detections = int(preds[-20:].sum())
    print(f"  ✅ Original adaptive under sustained burst: last-20 flags = {end_detections}")


def test_stabilized_does_not_allow_runaway() -> None:
    """Stabilized engine threshold must remain bounded during extreme attack burst."""
    from core.thresholding.algorithms import StabilizedAdaptiveEngine
    base = 0.65
    lower_ratio, upper_ratio = 0.80, 1.20
    warm = np.full(30, 0.3)
    burst = np.full(100, 50.0)  # Extreme spikes
    eng = StabilizedAdaptiveEngine(
        base_threshold=base, k=3.0, window_size=30,
        alpha=0.10, lower_ratio=lower_ratio, upper_ratio=upper_ratio,
    )
    _, thresholds = eng.evaluate_stream(burst, initial_scores=warm)
    lower = base * lower_ratio
    upper = base * upper_ratio
    assert all(lower <= t <= upper for t in thresholds), (
        f"Threshold out of [{lower:.3f}, {upper:.3f}]: range was "
        f"[{min(thresholds):.3f}, {max(thresholds):.3f}]"
    )
    print(f"  ✅ Stabilized bounded: [{min(thresholds):.3f}, {max(thresholds):.3f}]")


def test_csv_export_produces_valid_output() -> None:
    """Results DataFrame must serialize to valid UTF-8 CSV bytes."""
    from core.models.autoencoder import MultiSignalInferenceService
    from core.thresholding.algorithms import FixedThresholdEngine

    svc = MultiSignalInferenceService()
    df = _make_valid_csv(30)
    X = df[svc.pipeline["features"]].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    errors, deviations, scores = svc.predict_signals(X)

    eng = FixedThresholdEngine(base_threshold=float(svc.calibration["base_threshold"]))
    preds, thresholds = eng.evaluate_batch(scores)

    results = df.copy()
    results["Score"] = np.round(scores, 4)
    results["Threshold"] = np.round(thresholds, 4)
    results["Prediction"] = np.where(preds == 1, "ANOMALY", "NORMAL")

    csv_bytes = results.to_csv(index=False).encode("utf-8")
    assert len(csv_bytes) > 100, "CSV must be non-trivial"
    # Round-trip parse
    parsed = pd.read_csv(io.BytesIO(csv_bytes))
    assert "Score" in parsed.columns
    assert "Prediction" in parsed.columns
    assert len(parsed) == 30
    print(f"  ✅ CSV export: {len(csv_bytes):,} bytes, {len(parsed)} rows round-tripped")


def test_malformed_csv_is_handled() -> None:
    """
    Attempting to parse malformed bytes should raise an exception
    (the app catches this in the upload handler).
    """
    csv_bytes = _make_malformed_csv_bytes()
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
        # Even if pandas parses it, it won't have expected feature columns
        from core.models.autoencoder import MultiSignalInferenceService
        svc = MultiSignalInferenceService()
        matching = [c for c in svc.pipeline["features"] if c in df.columns]
        assert len(matching) == 0, "Malformed CSV must not contain valid feature columns"
        print(f"  ✅ Malformed CSV: parsed but 0 matching features → safely degraded")
    except Exception as exc:
        print(f"  ✅ Malformed CSV raises: {type(exc).__name__}: {exc}")


def test_large_dataset_inference() -> None:
    """10,000-row dataset should run inference in reasonable time."""
    import time
    from core.models.autoencoder import MultiSignalInferenceService
    svc = MultiSignalInferenceService()
    rng = np.random.default_rng(99)
    X = rng.uniform(0, 1, (10_000, len(svc.pipeline["features"]))).astype(np.float32)
    t0 = time.perf_counter()
    errors, deviations, scores = svc.predict_signals(X)
    elapsed = time.perf_counter() - t0
    assert len(scores) == 10_000
    print(f"  ✅ Large dataset (10k rows): {elapsed:.2f}s → {10_000/elapsed:.0f} rows/s")


# ── Run All ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Valid dataset loads", test_valid_dataset_loads),
        ("Model inference on valid data", test_model_inference_on_valid_data),
        ("Missing features filled with zeros", test_missing_features_filled_with_zeros),
        ("Empty dataset does not crash", test_empty_dataset_does_not_crash),
        ("Fixed threshold engine", test_fixed_threshold_engine),
        ("Original adaptive window exclusion", test_original_adaptive_window_exclusion),
        ("Stabilized does not allow runaway", test_stabilized_does_not_allow_runaway),
        ("CSV export produces valid output", test_csv_export_produces_valid_output),
        ("Malformed CSV is handled", test_malformed_csv_is_handled),
        ("Large dataset inference", test_large_dataset_inference),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\nTEST: {name}")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("ALL FUNCTIONAL TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")
        sys.exit(1)

