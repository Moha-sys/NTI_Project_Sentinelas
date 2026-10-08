"""LightGBM root-cause classifier.

Given an anomalous telemetry window, predicts WHICH service is the root
cause (and which fault type produced it).
"""
import logging
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from app.ml.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

CLASSIFIERS = ("root_cause_service", "fault_type")


class RootCauseClassifier:
    """Two LightGBM multiclass heads: root_cause_service and fault_type."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, lgb.LGBMClassifier] = {}
        self.classes_: Dict[str, List[str]] = {}

    def fit(self, X: pd.DataFrame, y_root: pd.Series, y_fault: pd.Series) -> "RootCauseClassifier":
        if len(X) == 0:
            raise ValueError("Cannot train root-cause classifier on empty data")

        targets = {"root_cause_service": y_root, "fault_type": y_fault}
        for name, y in targets.items():
            model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=5,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=self.random_state,
                verbose=-1,
            )
            model.fit(X[FEATURE_NAMES], y)
            self.models[name] = model
            self.classes_[name] = [str(c) for c in model.classes_]
            train_acc = float(model.score(X[FEATURE_NAMES], y))
            logger.info("LightGBM '%s' fitted: %d classes, train acc %.3f",
                        name, len(self.classes_[name]), train_acc)
        return self

    def predict(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        self._check_fitted()
        return {
            name: [str(p) for p in model.predict(X[FEATURE_NAMES])]
            for name, model in self.models.items()
        }

    def predict_proba(self, X: pd.DataFrame, target: str) -> np.ndarray:
        self._check_fitted()
        return self.models[target].predict_proba(X[FEATURE_NAMES])

    def predict_window(self, X: pd.DataFrame) -> List[Dict]:
        """Full per-window RCA prediction with confidences for API consumption."""
        self._check_fitted()
        results = []
        for i in range(len(X)):
            window = X.iloc[[i]]
            entry: Dict = {"timestamp": X.index[i]}
            for name, model in self.models.items():
                proba = model.predict_proba(window[FEATURE_NAMES])[0]
                best = int(np.argmax(proba))
                entry[name] = str(model.classes_[best])
                entry[f"{name}_confidence"] = round(float(proba[best]), 4)
                entry[f"{name}_probabilities"] = {
                    str(c): round(float(p), 4)
                    for c, p in zip(model.classes_, proba)
                }
            results.append(entry)
        return results

    def feature_importance(self, target: str, top_n: int = 8) -> List[Dict]:
        self._check_fitted()
        importances = self.models[target].feature_importances_
        pairs = sorted(
            zip(FEATURE_NAMES, importances), key=lambda p: p[1], reverse=True
        )
        return [{"feature": f, "importance": int(i)} for f, i in pairs[:top_n]]

    def _check_fitted(self):
        if not self.models:
            raise RuntimeError("RootCauseClassifier is not fitted. Call fit() or load artifacts.")
