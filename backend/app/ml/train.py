"""Training entrypoint: fits IsolationForest + LightGBM RCA models.

Run:  python -m app.ml.train   (from backend/)

Evaluation uses a temporal split (first 70% of timestamps train, last 30%
test) to avoid leaking future windows into training.
"""
import json
import logging
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from app.ml.anomaly import AnomalyDetector
from app.ml.features import build_feature_matrix
from app.ml.rca import RootCauseClassifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app.ml.train")

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
DATASET_PATH = Path(__file__).resolve().parent / "data" / "aiops_chaos_dataset.csv"
MODEL_PATH = ARTIFACTS_DIR / "rca_pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

TRAIN_FRACTION = 0.7


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def train(dataset_path: Path = DATASET_PATH, artifacts_dir: Path = ARTIFACTS_DIR) -> Dict:
    df = load_dataset(dataset_path)
    logger.info("Loaded %d rows, %d timestamps", len(df), df["timestamp"].nunique())

    X, meta = build_feature_matrix(df)
    if meta is None:
        raise RuntimeError("Dataset is missing label columns")

    # Temporal split on timestamps
    sorted_ts = sorted(X.index)
    cut = int(len(sorted_ts) * TRAIN_FRACTION)
    train_ts, test_ts = set(sorted_ts[:cut]), set(sorted_ts[cut:])

    X_train, X_test = X.loc[list(train_ts)], X.loc[list(test_ts)]
    meta_train, meta_test = meta.loc[list(train_ts)], meta.loc[list(test_ts)]

    # --- Isolation Forest: fit on normal windows of the TRAIN period ---
    normal_train = X_train[meta_train["is_anomaly"] == 0]
    detector = AnomalyDetector(contamination=0.05)
    detector.fit(normal_train)

    # Anomaly detection quality on the TEST period
    test_flags = (meta_test["is_anomaly"] == 1).astype(int).values
    test_scores = detector.score(X_test)
    test_preds = (detector.predict(X_test) == -1).astype(int)
    try:
        det_auc = float(roc_auc_score(test_flags, -test_scores))
    except ValueError:
        det_auc = float("nan")
    det_acc = float((test_preds == test_flags).mean())
    logger.info("Anomaly detector on test: auc=%.4f acc=%.4f", det_auc, det_acc)

    # --- LightGBM RCA: train on ANOMALOUS windows of the TRAIN period ---
    anom_train_mask = meta_train["is_anomaly"] == 1
    X_rca_train = X_train[anom_train_mask]
    y_root_train = meta_train.loc[anom_train_mask, "root_cause_service"]
    y_fault_train = meta_train.loc[anom_train_mask, "fault_type"]

    if len(X_rca_train) < 10:
        raise RuntimeError(f"Too few anomalous training windows: {len(X_rca_train)}")

    # Hold out a slice of anomalous train windows for a sanity check
    X_rca_fit, X_rca_val, y_root_fit, y_root_val, y_fault_fit, y_fault_val = train_test_split(
        X_rca_train, y_root_train, y_fault_train,
        test_size=0.2, random_state=42, stratify=y_root_train,
    )

    classifier = RootCauseClassifier()
    classifier.fit(X_rca_fit, y_root_fit, y_fault_fit)

    val_root_acc = float(accuracy_score(y_root_val, classifier.predict(X_rca_val)["root_cause_service"]))
    val_fault_acc = float(accuracy_score(y_fault_val, classifier.predict(X_rca_val)["fault_type"]))

    # Full test-period evaluation on anomalous windows
    anom_test_mask = meta_test["is_anomaly"] == 1
    X_rca_test = X_test[anom_test_mask]
    y_root_test = meta_test.loc[anom_test_mask, "root_cause_service"]
    y_fault_test = meta_test.loc[anom_test_mask, "fault_type"]

    test_preds_root = classifier.predict(X_rca_test)["root_cause_service"]
    test_preds_fault = classifier.predict(X_rca_test)["fault_type"]
    test_root_acc = float(accuracy_score(y_root_test, test_preds_root))
    test_fault_acc = float(accuracy_score(y_fault_test, test_preds_fault))

    logger.info("RCA val root acc=%.4f fault acc=%.4f", val_root_acc, val_fault_acc)
    logger.info("RCA test root acc=%.4f fault acc=%.4f", test_root_acc, test_fault_acc)
    logger.info("\n%s", classification_report(y_root_test, test_preds_root, zero_division=0))

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"detector": detector, "classifier": classifier, "feature_names": list(X.columns)},
        MODEL_PATH,
    )

    metrics = {
        "dataset": str(dataset_path.name),
        "n_rows": int(len(df)),
        "n_timestamps": int(df["timestamp"].nunique()),
        "train_timestamps": len(train_ts),
        "test_timestamps": len(test_ts),
        "train_normal_windows": int(len(normal_train)),
        "train_anomaly_windows": int(len(X_rca_train)),
        "test_anomaly_windows": int(len(X_rca_test)),
        "anomaly_detector": {"test_auc": round(det_auc, 4), "test_accuracy": round(det_acc, 4)},
        "root_cause_lgbm": {
            "val_accuracy": round(val_root_acc, 4),
            "test_accuracy": round(test_root_acc, 4),
            "classes": classifier.classes_.get("root_cause_service", []),
        },
        "fault_type_lgbm": {
            "val_accuracy": round(val_fault_acc, 4),
            "test_accuracy": round(test_fault_acc, 4),
            "classes": classifier.classes_.get("fault_type", []),
        },
        "top_features_root_cause": classifier.feature_importance("root_cause_service"),
        "root_cause_classification_report": classification_report(
            y_root_test, test_preds_root, zero_division=0, output_dict=True
        ),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Artifacts saved to %s", artifacts_dir)
    return metrics


if __name__ == "__main__":
    result = train()
    print(json.dumps({k: v for k, v in result.items() if k != "root_cause_classification_report"}, indent=2))
