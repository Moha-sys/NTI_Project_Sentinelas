"""Tests for the ML RCA pipeline: features, Isolation Forest, LightGBM, API."""
import pandas as pd
import pytest

from app.ml.features import build_feature_matrix, FEATURE_NAMES, SERVICES
from app.ml.pipeline import get_rca_pipeline
from app.ml.train import DATASET_PATH, load_dataset


@pytest.fixture(scope="module")
def dataset():
    if not DATASET_PATH.exists():
        pytest.skip("chaos dataset not present")
    return load_dataset()


@pytest.fixture(scope="module")
def pipeline():
    return get_rca_pipeline().ensure_loaded()


def _window(df, is_anomaly, index=0):
    ts = sorted(df[df.is_anomaly == is_anomaly].timestamp.unique())[index]
    return df[df.timestamp == ts]


# --------------------------------------------------------------- features
def test_feature_matrix_shape_and_columns(dataset):
    X, meta = build_feature_matrix(dataset)
    assert list(X.columns) == FEATURE_NAMES
    assert len(X) == dataset["timestamp"].nunique()
    assert meta is not None
    assert "root_cause_service" in meta.columns
    assert not X.isna().any().any()


def test_feature_matrix_partial_snapshot_fills_missing_services(dataset):
    # Only frontend rows of a single timestamp: other services must be filled
    ts = dataset.timestamp.unique()[0]
    single_service = dataset[(dataset.service_name == "frontend") & (dataset.timestamp == ts)]
    X, _ = build_feature_matrix(single_service)
    assert len(X) == 1
    assert X.iloc[0]["cart-service__service_up"] == 1.0
    assert X.iloc[0]["redis-cart__latency_ms"] == 20.0


def test_feature_matrix_empty_raises():
    with pytest.raises(ValueError):
        build_feature_matrix(pd.DataFrame())


# ------------------------------------------------------- anomaly detection
def test_isolation_forest_flags_anomaly_and_normal(pipeline, dataset):
    anom = _window(dataset, 1, index=3)
    normal = _window(dataset, 0, index=10)

    r_anom = pipeline.analyze(anom.to_dict("records"))
    r_normal = pipeline.analyze(normal.to_dict("records"))

    assert r_anom["is_anomaly"] is True
    assert r_anom["anomaly_score"] < 0
    assert r_anom["root_cause_service"] is not None
    assert r_anom["root_cause_fault_type"] is not None

    assert r_normal["is_anomaly"] is False
    assert r_normal["root_cause_service"] is None


def test_anomaly_probability_bounded(pipeline, dataset):
    result = pipeline.analyze(_window(dataset, 1, index=0).to_dict("records"))
    assert 0.0 <= result["anomaly_probability"] <= 1.0


# -------------------------------------------------------- root cause model
@pytest.mark.parametrize("fault,expected_service", [
    ("cascade_redis_drop", "redis-cart"),
    ("pod_failure", "frontend"),
])
def test_root_cause_prediction_matches_label(dataset, fault, expected_service):
    subset = dataset[dataset.fault_type == fault]
    ts = subset.timestamp.unique()[0]
    window = dataset[dataset.timestamp == ts]
    label = window.root_cause_service.iloc[0]
    assert label == expected_service

    result = get_rca_pipeline().ensure_loaded().analyze(window.to_dict("records"))
    assert result["is_anomaly"] is True
    assert result["root_cause_service"] == label
    assert result["root_cause_fault_type"] == fault


def test_pipeline_health(pipeline):
    health = pipeline.health()
    assert health["loaded"] is True
    assert health["artifacts_exist"] is True


# ---------------------------------------------------------------- API
def test_ml_status_endpoint(test_client):
    resp = test_client.get("/api/v1/ml/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pipeline"]["loaded"] is True
    assert "anomaly_detector" in data["models"]


def test_ml_analyze_provided_rows(test_client, dataset):
    window = _window(dataset, 1, index=1)
    rows = window[[
        "timestamp", "service_name", "cpu_percent", "memory_mb",
        "latency_ms", "error_rate", "service_up",
    ]].to_dict("records")
    resp = test_client.post("/api/v1/ml/analyze", json={"rows": rows})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "provided_rows"
    analysis = data["analysis"]
    assert analysis["is_anomaly"] is True
    assert analysis["root_cause_service"] is not None


def test_ml_analyze_live_mesh(test_client):
    resp = test_client.post("/api/v1/ml/analyze", json={"rows": []})
    # Live mesh may be unavailable in some environments; accept 200 or 503
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        assert "analysis" in resp.json()
