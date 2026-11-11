"""
Traffic Analysis API Routes.
Provides feature distribution histograms, outlier boxplots, and Pearson correlation matrices.
"""

from fastapi import APIRouter, HTTPException, Query
import numpy as np
import pandas as pd
from backend.core.data_store import data_store, FEATURE_GROUPS

router = APIRouter(prefix="/traffic", tags=["Traffic Analysis"])

SIGNAL_GROUPS_LIST = [
    {
        "id": "volumetric",
        "name": "Packet Volumes",
        "category": "Volumetric",
        "description": "Forward & backward packet counts and forward packet length moments",
        "features": FEATURE_GROUPS.get("Packet Volumes", []),
    },
    {
        "id": "rates",
        "name": "Flow Rates",
        "category": "Velocity",
        "description": "Throughput velocity measuring byte rate and packet arrival velocity",
        "features": FEATURE_GROUPS.get("Flow Rates", []),
    },
    {
        "id": "timing",
        "name": "Timing & Inter-Arrival Time (IAT)",
        "category": "Temporal",
        "description": "Flow duration and statistical moments of packet inter-arrival intervals",
        "features": FEATURE_GROUPS.get("Timing & Inter-Arrival Time (IAT)", []),
    },
    {
        "id": "tcp_flags",
        "name": "TCP Flags & Headers",
        "category": "Protocol",
        "description": "Control flags (SYN, ACK, FIN, RST) and TCP sliding window configurations",
        "features": FEATURE_GROUPS.get("TCP Flags & Headers", []),
    },
    {
        "id": "behavior",
        "name": "Active & Idle Behavior",
        "category": "Behavioral",
        "description": "Periodic heartbeat and idle silence interval characterization",
        "features": FEATURE_GROUPS.get("Active & Idle Behavior", []),
    },
]


@router.get("/groups")
async def get_feature_groups():
    """Returns the 5 orthogonal signal families formatted for the dashboard."""
    return {
        "groups": SIGNAL_GROUPS_LIST,
        "feature_groups": FEATURE_GROUPS,
    }


@router.get("/distribution")
async def get_feature_distribution(
    feature: str = Query(..., description="Feature column name"),
    bins: int = Query(25, description="Histogram bin count")
):
    """Calculates distribution histogram and Tukey fences for a selected feature."""
    if data_store.df is None or feature not in data_store.df.columns:
        # Fallback to first numerical column if not found
        num_cols = data_store.df.select_dtypes(include=[np.number]).columns if data_store.df is not None else []
        if len(num_cols) > 0:
            feature = num_cols[0]
        else:
            raise HTTPException(status_code=404, detail=f"Feature '{feature}' not found in dataset.")

    series = pd.to_numeric(data_store.df[feature], errors="coerce").dropna()
    series = series[np.isfinite(series)]

    if series.empty:
        return {
            "feature": feature,
            "bins": [],
            "fences": {
                "q1": 0, "median": 0, "q3": 0, "iqr": 0,
                "lower_fence": 0, "upper_fence": 0,
                "outlier_count": 0, "outlier_ratio": 0,
            }
        }

    counts, bin_edges = np.histogram(series, bins=bins)
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(counts))]

    # Outlier statistics (Tukey fences)
    q25 = float(series.quantile(0.25))
    median = float(series.median())
    q75 = float(series.quantile(0.75))
    iqr = q75 - q25
    lower_fence = float(max(series.min(), q25 - 1.5 * iqr))
    upper_fence = float(min(series.max(), q75 + 1.5 * iqr))
    outlier_count = int(((series < lower_fence) | (series > upper_fence)).sum())
    outlier_ratio = float(outlier_count / len(series)) if len(series) > 0 else 0.0

    return {
        "feature": feature,
        "bins": [{"bin": bin_labels[i], "count": int(counts[i])} for i in range(len(counts))],
        "fences": {
            "q1": round(q25, 4),
            "median": round(median, 4),
            "q3": round(q75, 4),
            "iqr": round(iqr, 4),
            "lower_fence": round(lower_fence, 4),
            "upper_fence": round(upper_fence, 4),
            "outlier_count": outlier_count,
            "outlier_ratio": round(outlier_ratio, 4),
        }
    }


@router.get("/correlation")
async def get_feature_correlation(
    features: str = Query(None, description="Comma-separated list of features")
):
    """Computes Pearson correlation matrix across selected features."""
    if data_store.df is None or data_store.df.empty:
        raise HTTPException(status_code=400, detail="Dataset not loaded.")

    if features:
        feat_list = [f.strip() for f in features.split(",") if f.strip() in data_store.df.columns]
    else:
        # Default top 8 numerical features
        num_cols = list(data_store.df.select_dtypes(include=[np.number]).columns)
        feat_list = [c for c in num_cols if not c.lower().endswith("label")][:8]

    if not feat_list:
        raise HTTPException(status_code=400, detail="No valid numerical features for correlation.")

    sub_df = data_store.df[feat_list].apply(pd.to_numeric, errors="coerce").fillna(0)
    corr_matrix = sub_df.corr().fillna(0).round(3).values.tolist()

    return {
        "features": feat_list,
        "matrix": corr_matrix,
    }
