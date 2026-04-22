"""Evaluation: metrics, calibration, and fairness audit.

Produces a metrics report at models/eval_report.json and a reliability
diagram plot at models/reliability.png.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from agri_credit_score.config import ALL_FEATURES, CATEGORICAL_FEATURES, TARGET


def _expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> float:
    """Compute ECE with uniform binning."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _ks_statistic(y_true: np.ndarray, probs: np.ndarray) -> float:
    """KS statistic between score distributions for positives and negatives."""
    from scipy.stats import ks_2samp

    pos = probs[y_true == 1]
    neg = probs[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(ks_2samp(pos, neg).statistic)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Primary and secondary metrics."""
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    top_decile = probs >= np.quantile(probs, 0.9)
    base_rate = y_true.mean()
    lift = (y_true[top_decile].mean() / base_rate) if base_rate > 0 else 0.0

    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": _expected_calibration_error(y_true, probs),
        "ks": _ks_statistic(y_true, probs),
        "lift_top_decile": float(lift),
        "confusion_at_0.5": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "n": int(len(y_true)),
        "base_rate": float(base_rate),
    }


def fairness_audit(
    df: pd.DataFrame, probs: np.ndarray, groups: list[str]
) -> dict:
    """Per-group AUC and positive-rate parity."""
    y = df[TARGET].to_numpy()
    results: dict[str, dict] = {}
    for group_col in groups:
        group_results = {}
        for value in df[group_col].dropna().unique():
            mask = (df[group_col] == value).to_numpy()
            if mask.sum() < 50 or y[mask].sum() < 5:
                continue
            group_results[str(value)] = {
                "n": int(mask.sum()),
                "default_rate": float(y[mask].mean()),
                "roc_auc": float(roc_auc_score(y[mask], probs[mask])),
                "mean_predicted_risk": float(probs[mask].mean()),
            }
        results[group_col] = group_results
    return results


@click.command()
@click.option("--model", required=True, type=click.Path(exists=True))
@click.option("--data", required=True, type=click.Path(exists=True))
@click.option("--out", default="models/eval_report.json")
@click.option("--seed", default=42)
def cli(model: str, data: str, out: str, seed: int) -> None:
    """Evaluate a trained model and write a detailed report."""
    with open(model, "rb") as f:
        clf = pickle.load(f)

    df = pd.read_parquet(data)
    _, df_test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=seed)

    X_test = df_test[ALL_FEATURES].copy()
    # Ensure categorical dtypes if loading XGBoost model
    for col in CATEGORICAL_FEATURES:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    y_test = df_test[TARGET].to_numpy()
    probs = clf.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, probs)
    fairness = fairness_audit(df_test, probs, groups=["gender", "region_code"])

    # Reliability data (for plotting externally)
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")

    report = {
        "overall_metrics": metrics,
        "fairness_audit": fairness,
        "calibration_curve": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        },
    }

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    click.echo(f"Overall metrics: ROC-AUC={metrics['roc_auc']:.3f} "
               f"PR-AUC={metrics['pr_auc']:.3f} "
               f"Brier={metrics['brier']:.3f} "
               f"ECE={metrics['ece']:.3f}")
    click.echo(f"Report written to {out_path}")


if __name__ == "__main__":
    cli()
