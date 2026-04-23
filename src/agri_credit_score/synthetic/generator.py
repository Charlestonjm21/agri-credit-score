"""Synthetic borrower dataset generator.

Produces a dataset calibrated to plausible Tanzanian smallholder-lending
base rates. A signal ceiling (see MAX_ACHIEVABLE_AUC_NOISE) is enforced by
adding irreducible noise to the label-generation process so that models
which exceed it are clearly overfitting to the generator.

Usage
-----
CLI:
    python -m agri_credit_score.synthetic.generator --n 10000 --out data/synthetic/borrowers.parquet

Python:
    from agri_credit_score.synthetic.generator import generate
    df = generate(n=10_000, seed=42)
"""
from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd

from agri_credit_score.synthetic.archetypes import (
    ALL_REGIONS,
    ARCHETYPES,
    CROP_BY_REGION,
)

MAX_ACHIEVABLE_AUC_NOISE = 0.07  # fraction of label flip noise
SEASON_PHASES = ["planting", "growing", "harvest", "off_season"]
GENDERS = ["F", "M", "Other"]
GENDER_WEIGHTS = [0.52, 0.46, 0.02]  # reflects higher female smallholder participation


def _sample_normal_pos(mean: float, std: float, rng: np.random.Generator, size: int) -> np.ndarray:
    """Sample from a normal distribution clipped at 0 (for non-negative features)."""
    return np.clip(rng.normal(mean, std, size=size), 0, None)


def _sample_normal_bounded(
    mean: float,
    std: float,
    lo: float,
    hi: float,
    rng: np.random.Generator,
    size: int,
) -> np.ndarray:
    """Sample and clip to [lo, hi]."""
    return np.clip(rng.normal(mean, std, size=size), lo, hi)


def _generate_archetype(
    archetype, n: int, rng: np.random.Generator, as_of_date: pd.Timestamp
) -> pd.DataFrame:
    """Generate n borrowers of a single archetype."""
    regions = rng.choice(archetype.regions or ALL_REGIONS, size=n)
    primary_crops = np.array(
        [rng.choice(CROP_BY_REGION[r]) for r in regions]
    )

    df = pd.DataFrame(
        {
            "borrower_id": [f"B{rng.integers(1e9, 1e10)}" for _ in range(n)],
            "as_of_date": as_of_date,
            "archetype": archetype.name,
            "region_code": regions,
            "primary_crop": primary_crops,
            "season_phase": rng.choice(SEASON_PHASES, size=n, p=[0.3, 0.35, 0.2, 0.15]),
            # Mobile money
            "mm_tx_count_90d": _sample_normal_pos(*archetype.mm_tx_count_90d, rng, n).astype(int),
            "mm_inflow_total_90d": _sample_normal_pos(*archetype.mm_inflow_total_90d, rng, n),
            "mm_outflow_total_90d": _sample_normal_pos(*archetype.mm_outflow_total_90d, rng, n),
            "mm_balance_mean_90d": _sample_normal_pos(*archetype.mm_balance_mean_90d, rng, n),
            "mm_balance_cv_90d": _sample_normal_bounded(
                *archetype.mm_balance_cv_90d, 0, 3, rng, n
            ),
            "mm_unique_counterparties_90d": _sample_normal_pos(
                *archetype.mm_unique_counterparties_90d, rng, n
            ).astype(int),
            "mm_recurrence_score": _sample_normal_bounded(
                *archetype.mm_recurrence_score, 0, 1, rng, n
            ),
            "mm_night_tx_ratio": _sample_normal_bounded(
                *archetype.mm_night_tx_ratio, 0, 1, rng, n
            ),
            "mm_weekend_tx_ratio": _sample_normal_bounded(
                *archetype.mm_weekend_tx_ratio, 0, 1, rng, n
            ),
            "mm_days_since_last_tx": _sample_normal_pos(
                *archetype.mm_days_since_last_tx, rng, n
            ).astype(int),
            "mm_active_days_90d": _sample_normal_bounded(
                *archetype.mm_active_days_90d, 0, 90, rng, n
            ).astype(int),
            # Agri context
            "rainfall_anomaly_30d": rng.normal(
                archetype.rainfall_anomaly_90d[0] * 0.8,
                archetype.rainfall_anomaly_90d[1],
                size=n,
            ),
            "rainfall_anomaly_90d": rng.normal(*archetype.rainfall_anomaly_90d, size=n),
            "ndvi_current": _sample_normal_bounded(*archetype.ndvi_current, 0, 1, rng, n),
            "ndvi_trend_30d": rng.normal(0, 0.05, size=n),
            "commodity_price_trend_30d": rng.normal(0.01, 0.08, size=n),
            "commodity_price_volatility_30d": _sample_normal_pos(0.1, 0.05, rng, n),
            "days_to_harvest": rng.integers(0, 180, size=n),
            # Borrower
            "age": _sample_normal_bounded(38, 12, 18, 75, rng, n).astype(int),
            "gender": rng.choice(GENDERS, size=n, p=GENDER_WEIGHTS),
            "years_farming": _sample_normal_pos(*archetype.years_farming, rng, n).astype(int),
            "prior_loans_count": _sample_normal_pos(
                *archetype.prior_loans_count, rng, n
            ).astype(int),
            # Loan
            "loan_amount_tzs": _sample_normal_pos(600_000, 300_000, rng, n),
            "loan_term_days": rng.choice([90, 120, 180, 270, 365], size=n),
        }
    )

    # Derived feature
    df["mm_net_flow_90d"] = df["mm_inflow_total_90d"] - df["mm_outflow_total_90d"]
    df["base_default_rate"] = archetype.base_default_rate
    return df


def _generate_labels(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Generate default labels with realistic feature-label relationships plus noise.

    The label probability combines:
    - The archetype base rate (dominant signal)
    - Small perturbations from individual feature values (realistic noise in the signal)
    - Label flip noise (irreducible)
    """
    p = df["base_default_rate"].to_numpy().copy()

    # Feature-driven perturbations (stronger signal than archetype-only)
    p += 0.25 * (df["mm_balance_cv_90d"].to_numpy() - 0.7)
    p += 0.15 * (-df["mm_active_days_90d"].to_numpy() / 90 + 0.5)
    p += 0.20 * (-df["rainfall_anomaly_90d"].to_numpy())
    p += 0.10 * (df["mm_night_tx_ratio"].to_numpy() - 0.1)
    p += 0.10 * (-df["mm_recurrence_score"].to_numpy() + 0.3)
    p += 0.05 * (df["mm_days_since_last_tx"].to_numpy() / 30)
    p += 0.08 * (-df["ndvi_current"].to_numpy() + 0.5)

    p = np.clip(p, 0.01, 0.95)
    labels = (rng.random(len(p)) < p).astype(int)

    # Irreducible label noise - caps achievable AUC
    flip_mask = rng.random(len(labels)) < MAX_ACHIEVABLE_AUC_NOISE
    labels[flip_mask] = 1 - labels[flip_mask]

    return labels


def _inject_missingness(
    df: pd.DataFrame, rate: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Inject MCAR missingness at the given rate across numeric feature columns."""
    from agri_credit_score.config import (
        AGRI_NUMERIC_FEATURES,
        BORROWER_NUMERIC_FEATURES,
        MOBILE_MONEY_FEATURES,
    )

    missable = MOBILE_MONEY_FEATURES + AGRI_NUMERIC_FEATURES + BORROWER_NUMERIC_FEATURES
    for col in missable:
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan
    return df


def generate(
    n: int = 10_000,
    seed: int = 42,
    missingness_rate: float = 0.05,
    as_of_date: str = "2026-04-22",
    spread_months: int = 0,
) -> pd.DataFrame:
    """Generate a synthetic borrower dataset.

    Parameters
    ----------
    n : number of borrowers
    seed : RNG seed for reproducibility
    missingness_rate : fraction of numeric values to set NaN (MCAR)
    as_of_date : scoring reference date (ISO string); used as the end anchor
    spread_months : if > 0, distribute borrowers across this many monthly
        buckets ending at as_of_date with linearly increasing weights
        (more recent months get proportionally more rows). Required for
        time-based holdout evaluation in train.py.

    Returns
    -------
    pd.DataFrame with features plus a 'defaulted' integer target column.
    """
    rng = np.random.default_rng(seed)
    as_of = pd.Timestamp(as_of_date)

    shares = np.array([a.population_share for a in ARCHETYPES])
    shares = shares / shares.sum()
    counts = rng.multinomial(n, shares)

    frames = [
        _generate_archetype(arch, int(c), rng, as_of)
        for arch, c in zip(ARCHETYPES, counts, strict=True)
        if c > 0
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df["defaulted"] = _generate_labels(df, rng)
    df = df.drop(columns=["base_default_rate"])
    df = _inject_missingness(df, missingness_rate, rng)

    if spread_months > 0:
        # Compute monthly first-of-month dates from oldest to newest,
        # ending at the month containing as_of_date.
        month_starts = [
            (as_of - pd.DateOffset(months=i)).replace(day=1)
            for i in range(spread_months - 1, -1, -1)
        ]
        # Linear weights: month 1 weight = 1, month N weight = N.
        weights = np.arange(1, spread_months + 1, dtype=float)
        weights /= weights.sum()
        chosen_idx = rng.choice(spread_months, size=len(df), p=weights)
        month_arr = np.array(month_starts, dtype="datetime64[ns]")
        df["as_of_date"] = month_arr[chosen_idx]

    return df


@click.command()
@click.option("--n", default=10_000, help="Number of borrowers to generate.")
@click.option("--seed", default=42, help="Random seed.")
@click.option("--missingness", default=0.05, help="MCAR missingness rate.")
@click.option("--out", default="data/synthetic/borrowers.parquet", help="Output path.")
@click.option("--spread-months", default=0, help="Distribute borrowers across N monthly cohorts (0 = single as_of_date).")
def cli(n: int, seed: int, missingness: float, out: str, spread_months: int) -> None:
    """Generate synthetic borrower data and write to parquet."""
    df = generate(n=n, seed=seed, missingness_rate=missingness, spread_months=spread_months)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    click.echo(f"Wrote {len(df):,} borrowers to {out_path}")
    click.echo(f"Default rate: {df['defaulted'].mean():.3f}")
    click.echo(f"Archetype distribution:\n{df['archetype'].value_counts()}")


if __name__ == "__main__":
    cli()
