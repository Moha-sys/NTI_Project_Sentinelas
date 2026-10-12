"""Isolation Forest anomaly detector over wide-format telemetry snapshots."""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.ml.features import FEATURE_NAMES

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Wraps sklearn IsolationForest trained on healthy telemetry snapshots.

    The detector is fit ONLY on normal baseline windows, so any score below
    zero marks a deviation from steady-state behavior.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.baseline_scores_: Optional[np.ndarray] = None

    def fit(self, X_normal: pd.DataFrame) -> "AnomalyDetector":
        if X_normal.empty:
            raise ValueError("Cannot fit anomaly detector on empty baseline")
        self.model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            max_samples="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_normal[FEATURE_NAMES])
        self.baseline_scores_ = self.model.decision_function(X_normal[FEATURE_NAMES])
        logger.info(
            "IsolationForest fitted on %d normal windows (mean score %.4f)",
            len(X_normal),
            float(self.baseline_scores_.mean()),
        )
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Raw decision function: negative = anomaly."""
        self._check_fitted()
        return self.model.decision_function(X[FEATURE_NAMES])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """+1 = normal, -1 = anomaly (sklearn convention)."""
        self._check_fitted()
        return self.model.predict(X[FEATURE_NAMES])

    def predict_dict(self, X: pd.DataFrame) -> List[Dict]:
        """Per-window anomaly report with scores for API consumption."""
        scores = self.score(X)
        flags = self.predict(X)
        reports = []
        for ts, score, flag in zip(X.index, scores, flags):
            is_anomaly = bool(flag == -1)
            try:
                ts_value = int(ts)
            except (TypeError, ValueError):
                ts_value = ts
            reports.append({
                "timestamp": ts_value,
                "is_anomaly": is_anomaly,
                "anomaly_score": round(float(score), 4),
                # Normalized 0..1 where 1 = most anomalous
                "anomaly_probability": round(float(1.0 / (1.0 + np.exp(score * 4))), 4),
            })
        return reports

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("AnomalyDetector is not fitted. Call fit() or load artifacts.")
