"""Inference pipeline: telemetry rows -> anomaly verdict -> root cause.

Loads trained artifacts lazily; trains them on first use if missing so the
API works out of the box.
"""
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from app.ml.anomaly import AnomalyDetector
from app.ml.features import build_feature_matrix, rows_to_dataframe
from app.ml.rca import RootCauseClassifier

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "rca_pipeline.joblib"


class RCAPipeline:
    """Anomaly detection + root-cause classification over telemetry windows."""

    def __init__(self):
        self.detector: Optional[AnomalyDetector] = None
        self.classifier: Optional[RootCauseClassifier] = None
        self.loaded_from: Optional[str] = None

    # ------------------------------------------------------------------ load
    def load(self) -> "RCAPipeline":
        if MODEL_PATH.exists():
            import joblib
            payload = joblib.load(MODEL_PATH)
            self.detector = payload["detector"]
            self.classifier = payload["classifier"]
            self.loaded_from = str(MODEL_PATH)
            logger.info("RCA pipeline loaded from %s", MODEL_PATH)
            return self

        logger.info("No artifacts found at %s - training from dataset", MODEL_PATH)
        from app.ml.train import train
        train()
        return self.load()

    def ensure_loaded(self) -> "RCAPipeline":
        if self.detector is None or self.classifier is None:
            self.load()
        return self

    # --------------------------------------------------------------- predict
    def analyze(self, rows: List[Dict]) -> Dict:
        """Full analysis of a telemetry snapshot (list of long-format rows).

        Returns anomaly verdict plus root-cause prediction when anomalous.
        """
        self.ensure_loaded()
        started = time.time()

        df = rows_to_dataframe(rows)
        X, _ = build_feature_matrix(df)
        anomaly_report = self.detector.predict_dict(X)

        windows = []
        for i, ts in enumerate(X.index):
            window = X.iloc[[i]]
            entry = dict(anomaly_report[i])
            if entry["is_anomaly"]:
                rca = self.classifier.predict_window(window)[0]
                rca.pop("timestamp", None)
                entry["root_cause"] = rca
                entry["top_features"] = self.classifier.feature_importance(
                    "root_cause_service", top_n=6
                )
            else:
                entry["root_cause"] = None
            windows.append(entry)

        is_anomaly = any(w["is_anomaly"] for w in windows)
        worst = min(windows, key=lambda w: w["anomaly_score"])

        # Consensus root cause across anomalous windows (majority vote)
        root_cause = None
        if is_anomaly:
            votes: Dict[str, int] = {}
            for w in windows:
                if w["is_anomaly"] and w.get("root_cause"):
                    rc = w["root_cause"]["root_cause_service"]
                    votes[rc] = votes.get(rc, 0) + 1
            if votes:
                root_cause = max(votes, key=votes.get)

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": worst["anomaly_score"],
            "anomaly_probability": worst["anomaly_probability"],
            "root_cause_service": root_cause,
            "root_cause_fault_type": (
                next(
                    (w["root_cause"]["fault_type"] for w in windows
                     if w["is_anomaly"] and w.get("root_cause")
                     and w["root_cause"]["root_cause_service"] == root_cause),
                    None,
                )
                if root_cause else None
            ),
            "windows": windows,
            "analysis_ms": round((time.time() - started) * 1000, 2),
        }

    def health(self) -> Dict:
        return {
            "loaded": self.detector is not None and self.classifier is not None,
            "source": self.loaded_from,
            "artifacts_exist": MODEL_PATH.exists(),
        }


_pipeline: Optional[RCAPipeline] = None


def get_rca_pipeline() -> RCAPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RCAPipeline()
    return _pipeline
