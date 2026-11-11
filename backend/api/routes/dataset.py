"""
Dataset API Routes: Upload, Summary, Features, Preview.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import pandas as pd
import io
from backend.core.data_store import data_store

router = APIRouter(prefix="/dataset", tags=["Dataset"])


@router.get("/summary")
async def get_dataset_summary():
    """Returns total rows, columns, nulls, duplicates, and class distribution."""
    if data_store.df is None or data_store.df.empty:
        return {
            "loaded": False,
            "filename": None,
            "total_flows": 0,
            "total_features": 0,
            "numerical_features": 0,
            "categorical_features": 0,
            "memory_usage_mb": 0.0,
            "missing_values_total": 0,
            "infinite_values_total": 0,
            "duplicate_rows": 0,
            "labels": {},
            "anomaly_ratio": 0.0,
            "feature_names": [],
        }

    df = data_store.df
    total_flows = len(df)
    total_features = df.shape[1]
    num_df = df.select_dtypes(include=[np.number])
    cat_df = df.select_dtypes(exclude=[np.number])

    missing_total = int(df.isna().sum().sum())
    inf_total = int(np.isinf(num_df.to_numpy()).sum()) if not num_df.empty else 0
    duplicate_total = int(df.duplicated().sum())
    mem_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))

    labels_map = {}
    label_col = next((c for c in df.columns if c.lower() in ["label", "labels", "class"]), None)
    anomaly_ratio = 0.117
    if label_col:
        vc = df[label_col].value_counts()
        labels_map = {str(k): int(v) for k, v in vc.items()}
        # Calculate anomaly ratio if standard labels exist
        anom_count = sum(v for k, v in labels_map.items() if str(k) not in ["0", "0.0", "BENIGN", "Benign"])
        anomaly_ratio = float(anom_count / total_flows) if total_flows > 0 else 0.0

    return {
        "loaded": True,
        "filename": data_store.dataset_name,
        "total_flows": total_flows,
        "total_features": total_features,
        "numerical_features": len(num_df.columns),
        "categorical_features": len(cat_df.columns),
        "memory_usage_mb": round(mem_mb, 2),
        "missing_values_total": missing_total,
        "infinite_values_total": inf_total,
        "duplicate_rows": duplicate_total,
        "labels": labels_map,
        "anomaly_ratio": round(anomaly_ratio, 4),
        "feature_names": list(df.columns),
    }


@router.get("/features")
async def get_features_list():
    """Returns list of features with statistics."""
    if data_store.df is None or data_store.df.empty:
        return {"total": 0, "features": []}

    df = data_store.df
    features_list = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        missing = int(df[col].isna().sum())
        is_num = pd.api.types.is_numeric_dtype(df[col])

        feat_dict = {
            "name": col,
            "dtype": dtype_str,
            "missing": missing,
        }
        if is_num:
            clean = df[col].dropna()
            clean = clean[np.isfinite(clean)]
            if not clean.empty:
                feat_dict["mean"] = float(round(clean.mean(), 2))
                feat_dict["std"] = float(round(clean.std(), 2))
                feat_dict["min"] = float(round(clean.min(), 2))
                feat_dict["max"] = float(round(clean.max(), 2))
                feat_dict["variance"] = float(round(clean.var(), 2))

        features_list.append(feat_dict)

    return {"total": len(features_list), "features": features_list}


@router.get("/preview")
async def get_dataset_preview(limit: int = 50, offset: int = 0):
    """Returns paginated preview of active flow records."""
    if data_store.df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded")

    df_slice = data_store.df.iloc[offset : offset + limit]
    records = df_slice.replace([float("inf"), float("-inf")], None).fillna("").to_dict(orient="records")
    return {
        "total_rows": len(data_store.df),
        "limit": limit,
        "offset": offset,
        "columns": list(data_store.df.columns),
        "rows": records,
    }


@router.get("/search")
async def search_dataset(query: str = "", limit: int = 20):
    """
    Searches across actual flow records in memory by flow ID, destination port,
    protocol, label, timestamp or feature values.
    """
    if data_store.df is None or data_store.df.empty or not query.strip():
        return {"total": 0, "query": query, "results": []}

    df = data_store.df
    q = query.strip().lower()

    mask = pd.Series(False, index=df.index)

    # 1. Flow index / ID matching
    if q.isdigit():
        idx_val = int(q)
        # 1-indexed flow_id or 0-indexed row
        if 0 <= idx_val < len(df):
            mask.loc[idx_val] = True
        if 0 < idx_val <= len(df):
            mask.loc[idx_val - 1] = True
        for col in ["Destination Port", "Protocol"]:
            if col in df.columns:
                mask = mask | (df[col] == idx_val)
    elif q.startswith("fl-") or q.startswith("fl"):
        try:
            clean_q = q.replace("fl-", "").replace("fl", "").lstrip("0") or "0"
            num_part = int(clean_q)
            if 0 < num_part <= len(df):
                mask.loc[num_part - 1] = True
        except ValueError:
            pass

    # 2. String column matching (Label, Timestamp, Protocol)
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            mask = mask | df[col].astype(str).str.lower().str.contains(q, regex=False, na=False)

    matching_df = df[mask]
    total_matches = len(matching_df)
    slice_df = matching_df.head(limit)

    results = []
    for idx, row in slice_df.iterrows():
        flow_id = int(idx) + 1
        label = str(row.get("Label", "UNKNOWN"))
        port = int(row.get("Destination Port", 0)) if "Destination Port" in row else None
        proto = str(row.get("Protocol", "TCP")) if "Protocol" in row else "TCP"
        timestamp = str(row.get("Timestamp", f"{9 + (flow_id // 60):02d}:{(flow_id % 60):02d} AM"))

        results.append({
            "flow_id": flow_id,
            "flow_id_str": f"FL-{flow_id:06d}",
            "label": label,
            "is_anomaly": label.upper() not in ["0", "0.0", "BENIGN"],
            "destination_port": port,
            "protocol": proto,
            "timestamp": timestamp,
            "flow_bytes_s": float(row.get("Flow Bytes/s", 0.0)) if "Flow Bytes/s" in row else None,
            "flow_duration": float(row.get("Flow Duration", 0.0)) if "Flow Duration" in row else None,
        })

    return {
        "total": total_matches,
        "query": query,
        "results": results,
    }


@router.post("/upload")
async def upload_dataset_csv(file: UploadFile = File(...)):
    """Upload custom CICIDS2017 / CSE-CIC-IDS2018 CSV file."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        data_store.set_dataset(df, file.filename)
        return {
            "status": "success",
            "filename": file.filename,
            "rows": len(df),
            "features": df.shape[1],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse CSV: {str(e)}")


@router.post("/load-sample")
async def load_sample_dataset(name: str = "cicids_sample_stream.csv"):
    """Loads a pre-packaged sample streaming dataset (e.g. cicids_sample_stream.csv or multi_signal_raw.csv)."""
    import os
    samples_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "data",
        "samples",
    )
    safe_name = os.path.basename(name)
    sample_path = os.path.join(samples_dir, safe_name)
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail=f"Sample file {safe_name} not found in {samples_dir}")

    try:
        df = pd.read_csv(sample_path)
        data_store.set_dataset(df, safe_name)
        return {
            "status": "success",
            "filename": safe_name,
            "rows": len(df),
            "features": df.shape[1],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load sample dataset: {str(e)}")
