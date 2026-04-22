"""Train XGBoost scoring model and logistic regression baseline.

Ships two artifacts to the output directory:
    xgb.pkl            - fitted XGBoost model + calibrator
    lr.pkl             - fitted logistic regression baseline (interpretability anchor)
    metadata.json      - training metadata (date, metrics, feature list)

The training pipeline:
    1. Train/calibration/test split
    2. Optuna hyperparameter search on training fold (5-fold stratified CV)
    3. Final fit on training + validation
    4. Isotonic calibration on a held-out calibration slice
    5. Evaluation on the test fold

XGBoost handles missing values natively. Logistic regression uses the
preprocessing pipeline in agri_credit_score.features.pipeline.
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from agri_credit_score.config import ALL_FEATURES, MODEL_VERSION, TARGET
from agri_credit_score.features.pipeline import build_preprocessor

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _prep_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categoricals to XGBoost-compatible categorical dtype."""
    from agri_credit_score.config import CATEGORICAL_FEATURES

    X = df[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


def _optuna_search(
    X_train: pd.DataFrame, y_train: np.ndarray, n_trials: int = 30, seed: int = 42
) -> dict:
    """Run Optuna hyperparameter search with 5-fold stratified CV on PR-AUC."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "enable_categorical": True,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "random_state": seed,
        }
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        params["scale_pos_weight"] = neg / max(pos, 1)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train[train_idx]
            y_val = y_train[val_idx]
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, verbose=False)
            preds = model.predict_proba(X_val)[:, 1]
            scores.append(average_precision_score(y_val, preds))
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_xgboost(
    df: pd.DataFrame, n_trials: int = 30, seed: int = 42
) -> tuple[CalibratedClassifierCV, dict]:
    """Train XGBoost with Optuna search, then isotonically calibrate."""
    X = _prep_xgb_features(df)
    y = df[TARGET].to_numpy()

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_dev, y_dev, test_size=0.15, stratify=y_dev, random_state=seed
    )

    click.echo(f"  Search space: {n_trials} trials, 5-fold CV on PR-AUC...")
    best_params = _optuna_search(X_train, y_train, n_trials=n_trials, seed=seed)
    click.echo(f"  Best params: {best_params}")

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "enable_categorical": True,
        "scale_pos_weight": neg / max(pos, 1),
        "random_state": seed,
    }
    final_params = {**base_params, **best_params}

    base_model = xgb.XGBClassifier(**final_params)
    base_model.fit(X_train, y_train, verbose=False)

    # Isotonic calibration on held-out calibration slice
    frozen = FrozenEstimator(base_model)
    calibrated = CalibratedClassifierCV(frozen, method="isotonic")
    calibrated.fit(X_cal, y_cal)

    # Evaluate on test set
    test_probs = calibrated.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_probs)),
        "pr_auc": float(average_precision_score(y_test, test_probs)),
        "brier": float(brier_score_loss(y_test, test_probs)),
        "test_size": int(len(y_test)),
        "test_default_rate": float(y_test.mean()),
    }
    return calibrated, {"best_params": best_params, "test_metrics": metrics}


def train_logistic_baseline(
    df: pd.DataFrame, seed: int = 42
) -> tuple[Pipeline, dict]:
    """Train a logistic regression baseline using the preprocessing pipeline."""
    X = df[ALL_FEATURES]
    y = df[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    preprocessor = build_preprocessor()
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=seed
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "pr_auc": float(average_precision_score(y_test, probs)),
        "brier": float(brier_score_loss(y_test, probs)),
    }
    return model, {"test_metrics": metrics}


@click.command()
@click.option("--data", required=True, type=click.Path(exists=True), help="Parquet data path")
@click.option("--out", default="models/", help="Output directory for artifacts")
@click.option("--trials", default=30, help="Optuna trial count")
@click.option("--seed", default=42)
def cli(data: str, out: str, trials: int, seed: int) -> None:
    """Train scoring models and persist artifacts."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading data from {data}...")
    df = pd.read_parquet(data)
    click.echo(f"  {len(df):,} rows, default rate {df[TARGET].mean():.3f}")

    click.echo("\nTraining logistic regression baseline...")
    lr_model, lr_info = train_logistic_baseline(df, seed=seed)
    click.echo(f"  LR test metrics: {lr_info['test_metrics']}")
    with open(out_dir / "lr.pkl", "wb") as f:
        pickle.dump(lr_model, f)

    click.echo("\nTraining XGBoost...")
    xgb_model, xgb_info = train_xgboost(df, n_trials=trials, seed=seed)
    click.echo(f"  XGB test metrics: {xgb_info['test_metrics']}")
    with open(out_dir / "xgb.pkl", "wb") as f:
        pickle.dump(xgb_model, f)

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": int(len(df)),
        "seed": seed,
        "xgboost": xgb_info,
        "logistic_regression": lr_info,
        "features": ALL_FEATURES,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    click.echo(f"\nArtifacts written to {out_dir}/")


if __name__ == "__main__":
    cli()
