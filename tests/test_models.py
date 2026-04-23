"""Tests for model training, evaluation, and explanation modules."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

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


def test_time_split_trains_and_predicts() -> None:
    """train_xgboost and train_logistic_baseline accept a pre-split df_test."""
    from agri_credit_score.synthetic.generator import generate

    df = generate(n=1000, seed=42, spread_months=24)
    df_sorted = df.sort_values("as_of_date").reset_index(drop=True)
    n_pool = int(len(df_sorted) * 0.8)
    df_pool = df_sorted.iloc[:n_pool].reset_index(drop=True)
    n_train = int(len(df_pool) * 0.8)
    df_train = df_pool.iloc[:n_train].reset_index(drop=True)
    df_test = df_pool.iloc[n_train:].reset_index(drop=True)

    xgb_model, xgb_info = train_xgboost(df_train, n_trials=3, seed=42, df_test=df_test)
    assert xgb_info["test_metrics"]["roc_auc"] > 0.5
    assert 0 <= xgb_info["test_metrics"]["brier"] <= 1

    lr_model, lr_info = train_logistic_baseline(df_train, seed=42, df_test=df_test)
    assert lr_info["test_metrics"]["roc_auc"] > 0.5


def test_time_split_both_metric_sets_comparable() -> None:
    """Both time-split and random-split metrics are computable from the same pool."""
    from agri_credit_score.synthetic.generator import generate

    df = generate(n=800, seed=42, spread_months=24)
    df_pool = df.sort_values("as_of_date").reset_index(drop=True)
    n_train = int(len(df_pool) * 0.8)
    df_time_train = df_pool.iloc[:n_train].reset_index(drop=True)
    df_time_test = df_pool.iloc[n_train:].reset_index(drop=True)

    df_rand_train, df_rand_test = train_test_split(
        df_pool, test_size=0.2, stratify=df_pool[TARGET], random_state=42
    )

    _, time_info = train_xgboost(df_time_train, n_trials=2, seed=42, df_test=df_time_test)
    _, rand_info = train_xgboost(df_rand_train, n_trials=2, seed=42, df_test=df_rand_test)

    for info in (time_info, rand_info):
        assert "test_metrics" in info
        m = info["test_metrics"]
        assert "roc_auc" in m and "pr_auc" in m and "brier" in m
        assert 0 <= m["roc_auc"] <= 1


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
