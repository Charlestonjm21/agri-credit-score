"""SHAP-based explainability for the XGBoost scoring model.

Provides per-prediction top-k feature attribution. Used by the API
to populate the `top_features` field of every score response.
"""
from __future__ import annotations

import pandas as pd
import shap

from agri_credit_score.config import FEATURE_LABELS


class Explainer:
    """SHAP wrapper that returns human-readable feature attributions."""

    def __init__(self, calibrated_xgb_model) -> None:
        # CalibratedClassifierCV(method="isotonic") fit on a FrozenEstimator
        # stores the frozen wrapper in calibrated_classifiers_[0].estimator
        # (modern sklearn). We need to reach through to the underlying XGB.
        underlying = self._unwrap(calibrated_xgb_model)
        self._explainer = shap.TreeExplainer(underlying)

    @staticmethod
    def _unwrap(calibrated):
        """Reach the raw XGBClassifier inside a CalibratedClassifierCV(FrozenEstimator(xgb))."""
        # Try calibrated_classifiers_ first (fitted CalibratedClassifierCV)
        if hasattr(calibrated, "calibrated_classifiers_") and calibrated.calibrated_classifiers_:
            cc = calibrated.calibrated_classifiers_[0]
            # cc.estimator is the frozen wrapper; .estimator on that is the raw model
            est = getattr(cc, "estimator", None)
            if est is not None:
                inner = getattr(est, "estimator", est)
                return inner
        # Fallbacks for older APIs
        est = getattr(calibrated, "estimator", None) or getattr(
            calibrated, "base_estimator", None
        )
        if est is not None:
            inner = getattr(est, "estimator", est)
            return inner
        return calibrated

    def explain(self, X: pd.DataFrame, top_k: int = 5) -> list[list[dict]]:
        """Return top-k SHAP attributions for each row in X.

        Each attribution is a dict with feature name, label, SHAP value,
        and direction ("increases_risk" or "decreases_risk").
        """
        shap_values = self._explainer.shap_values(X)
        results: list[list[dict]] = []

        for row_idx in range(len(X)):
            row_shap = shap_values[row_idx]
            order = sorted(range(len(row_shap)), key=lambda i: abs(row_shap[i]), reverse=True)
            top = []
            for i in order[:top_k]:
                feature = X.columns[i]
                value = float(row_shap[i])
                top.append(
                    {
                        "name": feature,
                        "label": FEATURE_LABELS.get(feature, feature),
                        "direction": "increases_risk" if value > 0 else "decreases_risk",
                        "shap_value": value,
                    }
                )
            results.append(top)

        return results
