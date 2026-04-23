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
    df: pd.DataFrame,
    n_trials: int = 30,
    seed: int = 42,
    df_test: pd.DataFrame | None = None,
) -> tuple[CalibratedClassifierCV, dict]:
    """Train XGBoost with Optuna search, then isotonically calibrate.

    Parameters
    ----------
    df : training pool. When df_test is None this is the full dataset and an
        internal random 80/20 split is used. When df_test is provided, df is
        treated as the training pool only and no internal test split is made.
    df_test : optional pre-split test set. Pass this when the caller controls
        the split strategy (e.g. time-based holdout).
    """
    X = _prep_xgb_features(df)
    y = df[TARGET].to_numpy()

    if df_test is None:
        X_dev, X_test, y_dev, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_dev, y_dev, test_size=0.15, stratify=y_dev, random_state=seed
        )
    else:
        X_test = _prep_xgb_features(df_test)
        y_test = df_test[TARGET].to_numpy()
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=seed
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
    df: pd.DataFrame,
    seed: int = 42,
    df_test: pd.DataFrame | None = None,
) -> tuple[Pipeline, dict]:
    """Train a logistic regression baseline using the preprocessing pipeline.

    Parameters
    ----------
    df : training pool (full dataset when df_test is None, training split otherwise).
    df_test : optional pre-split test set; mirrors the df_test contract in train_xgboost.
    """
    X = df[ALL_FEATURES]
    y = df[TARGET].to_numpy()

    if df_test is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )
    else:
        X_train, y_train = X, y
        X_test = df_test[ALL_FEATURES]
        y_test = df_test[TARGET].to_numpy()

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
@click.option(
    "--split-mode",
    type=click.Choice(["random", "time"]),
    default="random",
    show_default=True,
    help=(
        "'random': stratified random 80/20 split (default). "
        "'time': chronological split — requires as_of_date spread in the data "
        "(use spread_months > 0 in the generator). "
        "See docs/SPEC.md §6.2 for the full evaluation protocol."
    ),
)
def cli(data: str, out: str, trials: int, seed: int, split_mode: str) -> None:
    """Train scoring models and persist artifacts."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading data from {data}...")
    df = pd.read_parquet(data)
    click.echo(f"  {len(df):,} rows, default rate {df[TARGET].mean():.3f}")
    click.echo(f"  Split mode: {split_mode}")

    if split_mode == "random":
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
            "split_mode": "random",
            "n_train_rows": int(len(df)),
            "seed": seed,
            "xgboost": xgb_info,
            "logistic_regression": lr_info,
            "features": ALL_FEATURES,
        }

    else:  # split_mode == "time"
        if df["as_of_date"].nunique() == 1:
            click.echo(
                "WARNING: as_of_date has only one unique value. "
                "Time split is meaningless — regenerate data with spread_months > 0."
            )

        # Time-split evaluation protocol (see docs/SPEC.md §6.2):
        # Sort by as_of_date. The first 80% is the "pool" used for all model
        # training and within-pool comparison; the last 20% is a true future
        # holdout evaluated post-hoc on the primary (time-split) model.
        #
        # Within the pool two evaluations isolate the effect of split strategy
        # while holding data volume and distribution constant:
        #   time_split_metrics  — train on pool's first 80% (by date),
        #                         evaluate on pool's last 20% (by date)
        #   random_split_metrics — random 80/20 of pool, evaluate on 20%
        # Only the split strategy differs between the two; archetype mix and
        # sample size are identical.
        df_sorted = df.sort_values("as_of_date").reset_index(drop=True)
        n_total = len(df_sorted)
        n_pool = int(n_total * 0.8)

        df_pool = df_sorted.iloc[:n_pool].reset_index(drop=True)
        df_future = df_sorted.iloc[n_pool:].reset_index(drop=True)

        n_pool_train = int(len(df_pool) * 0.8)
        df_pool_time_train = df_pool.iloc[:n_pool_train].reset_index(drop=True)
        df_pool_time_test = df_pool.iloc[n_pool_train:].reset_index(drop=True)

        df_pool_rand_train, df_pool_rand_test = train_test_split(
            df_pool,
            test_size=0.2,
            stratify=df_pool[TARGET],
            random_state=seed,
        )

        # LR — time split
        click.echo("\nTraining logistic regression baseline (time split)...")
        lr_model, lr_time_info = train_logistic_baseline(
            df_pool_time_train, seed=seed, df_test=df_pool_time_test
        )
        click.echo(f"  LR time-split metrics: {lr_time_info['test_metrics']}")
        with open(out_dir / "lr.pkl", "wb") as f:
            pickle.dump(lr_model, f)

        # LR — random split (same pool, different selection)
        click.echo("\nTraining logistic regression baseline (random split)...")
        _, lr_rand_info = train_logistic_baseline(
            df_pool_rand_train, seed=seed, df_test=df_pool_rand_test
        )
        click.echo(f"  LR random-split metrics: {lr_rand_info['test_metrics']}")

        lr_info = {
            "time_split_metrics": lr_time_info["test_metrics"],
            "random_split_metrics": lr_rand_info["test_metrics"],
        }

        # XGBoost — time split (primary model, artifact saved)
        click.echo("\nTraining XGBoost (time split)...")
        xgb_model, xgb_time_info = train_xgboost(
            df_pool_time_train, n_trials=trials, seed=seed, df_test=df_pool_time_test
        )
        click.echo(f"  XGB time-split metrics: {xgb_time_info['test_metrics']}")
        with open(out_dir / "xgb.pkl", "wb") as f:
            pickle.dump(xgb_model, f)

        # XGBoost — random split (same pool, for comparison only)
        click.echo("\nTraining XGBoost (random split)...")
        _, xgb_rand_info = train_xgboost(
            df_pool_rand_train, n_trials=trials, seed=seed, df_test=df_pool_rand_test
        )
        click.echo(f"  XGB random-split metrics: {xgb_rand_info['test_metrics']}")

        # Evaluate primary model on true future holdout
        X_future = _prep_xgb_features(df_future)
        y_future = df_future[TARGET].to_numpy()
        future_probs = xgb_model.predict_proba(X_future)[:, 1]
        future_metrics = {
            "roc_auc": float(roc_auc_score(y_future, future_probs)),
            "pr_auc": float(average_precision_score(y_future, future_probs)),
            "brier": float(brier_score_loss(y_future, future_probs)),
            "n": int(len(y_future)),
        }
        click.echo(f"  XGB future-holdout metrics: {future_metrics}")

        xgb_info = {
            "best_params": xgb_time_info["best_params"],
            "time_split_metrics": xgb_time_info["test_metrics"],
            "random_split_metrics": xgb_rand_info["test_metrics"],
            "future_holdout_metrics": future_metrics,
        }

        metadata = {
            "model_version": MODEL_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "split_mode": "time",
            "n_total_rows": n_total,
            "n_pool_rows": int(len(df_pool)),
            "n_pool_train_rows": int(len(df_pool_time_train)),
            "n_future_holdout_rows": int(len(df_future)),
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
