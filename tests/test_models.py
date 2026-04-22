"""Tests for model training, evaluation, and explanation modules."""
from __future__ import annotations

import numpy as np

from agri_credit_score.config import ALL_FEATURES, CATEGORICAL_FEATURES, TARGET
from agri_credit_score.models.evaluate import (
    _expected_calibration_error,
    _ks_statistic,
    compute_metrics,
    fairness_audit,
)
from agri_credit_score.models.explain import Explainer
from agri_credit_score.models.train import train_logistic_baseline, train_xgboost
from agri_credit_score.synthetic.generator import generate


def test_logistic_baseline_trains_and_predicts() -> None:
    df = generate(n=800, seed=42)
    model, info = train_logistic_baseline(df, seed=42)
    metrics = info["test_metrics"]
    # Very loose bar - just checking it learned something above random
    assert metrics["roc_auc"] > 0.55, f"LR AUC too low: {metrics}"
    assert 0 <= metrics["brier"] <= 1


def test_xgboost_beats_random_with_few_trials() -> None:
    df = generate(n=1500, seed=42)
    model, info = train_xgboost(df, n_trials=5, seed=42)
    metrics = info["test_metrics"]
    assert metrics["roc_auc"] > 0.6, f"XGB AUC too low: {metrics}"

    # Verify predict_proba works on new data
    Xnew = df.head(20)[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        Xnew[col] = Xnew[col].astype("category")
    probs = model.predict_proba(Xnew)[:, 1]
    assert probs.shape == (20,)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_explainer_returns_top_k() -> None:
    df = generate(n=1000, seed=42)
    model, _ = train_xgboost(df, n_trials=5, seed=42)

    explainer = Explainer(model)
    X = df.head(3)[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    results = explainer.explain(X, top_k=5)
    assert len(results) == 3
    for attrs in results:
        assert len(attrs) == 5
        for a in attrs:
            assert set(a.keys()) >= {"name", "label", "direction", "shap_value"}
            assert a["direction"] in ("increases_risk", "decreases_risk")


def test_compute_metrics_sane() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    probs = rng.random(500)
    m = compute_metrics(y, probs)
    assert 0.3 < m["roc_auc"] < 0.7  # random-ish
    assert 0 <= m["brier"] <= 1
    assert 0 <= m["ece"] <= 1
    assert m["n"] == 500


def test_ece_zero_for_perfect_calibration() -> None:
    # If probs exactly equal the empirical mean, ECE should be near zero
    y = np.array([1] * 50 + [0] * 50)
    probs = np.array([0.5] * 100)
    ece = _expected_calibration_error(y, probs)
    assert ece < 0.05


def test_ks_statistic_bounds() -> None:
    y = np.array([0] * 100 + [1] * 100)
    probs = np.concatenate([np.full(100, 0.1), np.full(100, 0.9)])
    ks = _ks_statistic(y, probs)
    # Perfect separation means KS = 1
    assert ks > 0.95


def test_fairness_audit_returns_per_group() -> None:
    df = generate(n=2000, seed=42)
    rng = np.random.default_rng(0)
    # Synthetic probs correlated with true label
    probs = np.where(df[TARGET] == 1, rng.uniform(0.4, 1.0, len(df)), rng.uniform(0.0, 0.6, len(df)))
    result = fairness_audit(df, probs, groups=["gender"])
    assert "gender" in result
    for _group_val, stats in result["gender"].items():
        assert "roc_auc" in stats
        assert "default_rate" in stats
        assert 0 <= stats["roc_auc"] <= 1
