"""Feature engineering for chaos telemetry.

The raw telemetry is long-format: one row per (timestamp, service).
Models consume a wide-format vector per timestamp: per-service metrics
pivoted to columns plus cross-service aggregates.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SERVICES: List[str] = [
    "frontend",
    "cart-service",
    "payment-service",
    "productcatalog-service",
    "redis-cart",
]

METRICS: List[str] = ["cpu_percent", "memory_mb", "latency_ms", "error_rate", "service_up"]

# Fill values for services missing from a snapshot (match healthy steady-state)
DEFAULT_FILL: Dict[str, float] = {
    "cpu_percent": 2.0,
    "memory_mb": 80.0,
    "latency_ms": 20.0,
    "error_rate": 0.001,
    "service_up": 1.0,
}

SERVICE_FEATURES: List[str] = [f"{s}__{m}" for s in SERVICES for m in METRICS]

DERIVED_FEATURES: List[str] = [
    "latency_max",
    "latency_mean",
    "error_max",
    "error_mean",
    "cpu_max",
    "memory_max",
    "services_down",
    "services_down_ratio",
]

FEATURE_NAMES: List[str] = SERVICE_FEATURES + DERIVED_FEATURES

LABEL_COLUMNS: List[str] = ["is_anomaly", "fault_type", "root_cause_service"]


def rows_to_dataframe(rows: List[Dict]) -> pd.DataFrame:
    """Convert a list of telemetry row dicts into a DataFrame."""
    return pd.DataFrame(rows)


def build_feature_matrix(
    df: pd.DataFrame,
    timestamps: Optional[List] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Pivot long-format telemetry into one wide feature row per timestamp.

    Returns (X, meta) where meta holds the label columns when present.
    Missing services are filled with healthy defaults so partial snapshots
    (e.g. live mesh) still map onto the trained feature space.
    """
    if df.empty:
        raise ValueError("Cannot build features from an empty telemetry frame")

    work = df.copy()
    if "timestamp" not in work.columns:
        work["timestamp"] = 0

    records: List[Dict[str, float]] = []
    meta_records: List[Dict] = []
    for ts, group in work.groupby("timestamp", sort=True):
        feat: Dict[str, float] = {}
        for service in SERVICES:
            svc_rows = group[group["service_name"] == service]
            if svc_rows.empty:
                for metric in METRICS:
                    feat[f"{service}__{metric}"] = DEFAULT_FILL[metric]
                continue
            row = svc_rows.iloc[0]
            for metric in METRICS:
                value = row.get(metric, DEFAULT_FILL[metric])
                if pd.isna(value):
                    value = DEFAULT_FILL[metric]
                feat[f"{service}__{metric}"] = float(value)

        latencies = [feat[f"{s}__latency_ms"] for s in SERVICES]
        errors = [feat[f"{s}__error_rate"] for s in SERVICES]
        cpus = [feat[f"{s}__cpu_percent"] for s in SERVICES]
        memories = [feat[f"{s}__memory_mb"] for s in SERVICES]
        ups = [feat[f"{s}__service_up"] for s in SERVICES]
        down = sum(1 for u in ups if u < 0.5)

        feat["latency_max"] = float(np.max(latencies))
        feat["latency_mean"] = float(np.mean(latencies))
        feat["error_max"] = float(np.max(errors))
        feat["error_mean"] = float(np.mean(errors))
        feat["cpu_max"] = float(np.max(cpus))
        feat["memory_max"] = float(np.max(memories))
        feat["services_down"] = float(down)
        feat["services_down_ratio"] = down / len(SERVICES)

        records.append({"timestamp": ts, **feat})

        meta_rec: Dict = {"timestamp": ts}
        for col in LABEL_COLUMNS:
            if col in group.columns:
                values = group[col].dropna()
                meta_rec[col] = values.iloc[0] if len(values) else None
        meta_records.append(meta_rec)

    X = pd.DataFrame(records).set_index("timestamp")
    X = X.reindex(columns=FEATURE_NAMES)

    if timestamps is not None:
        X = X.loc[X.index.intersection(timestamps)]

    label_cols = [c for c in LABEL_COLUMNS if any(c in m for m in meta_records)]
    meta_df = pd.DataFrame(meta_records).set_index("timestamp") if meta_records else None
    if meta_df is not None and not label_cols:
        meta_df = None

    return X, meta_df
