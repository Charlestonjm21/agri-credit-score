"""Tests for the synthetic data generator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from agri_credit_score.config import ALL_FEATURES, TARGET
from agri_credit_score.synthetic.generator import generate


def test_generate_shape_and_columns() -> None:
    df = generate(n=500, seed=42)
    assert len(df) == 500
    for col in ALL_FEATURES + [TARGET]:
        assert col in df.columns, f"Missing column {col}"


def test_generate_reproducible_with_seed() -> None:
    df1 = generate(n=200, seed=7)
    df2 = generate(n=200, seed=7)
    # Compare stable numeric columns (borrower_id uses its own RNG draws)
    numeric_cols = [c for c in ALL_FEATURES if df1[c].dtype != "object"]
    pd.testing.assert_frame_equal(
        df1[numeric_cols].reset_index(drop=True),
        df2[numeric_cols].reset_index(drop=True),
    )


def test_default_rate_in_plausible_range() -> None:
    df = generate(n=5000, seed=42)
    rate = df[TARGET].mean()
    # Tanzanian smallholder loan default rates cluster 8-20%; with our
    # archetype mix + noise we expect ~15-30%.
    assert 0.08 <= rate <= 0.35, f"Implausible default rate: {rate:.3f}"


def test_missingness_injected() -> None:
    df = generate(n=1000, seed=42, missingness_rate=0.1)
    total_missing = df[ALL_FEATURES].isna().sum().sum()
    assert total_missing > 0, "Expected missing values with nonzero missingness rate"


def test_no_missingness_when_disabled() -> None:
    df = generate(n=500, seed=42, missingness_rate=0.0)
    # Categorical columns never had missingness anyway, numerics should be clean
    from agri_credit_score.config import NUMERIC_FEATURES

    assert df[NUMERIC_FEATURES].isna().sum().sum() == 0


def test_balance_cv_bounded() -> None:
    df = generate(n=500, seed=42, missingness_rate=0.0)
    assert (df["mm_balance_cv_90d"] >= 0).all()
    assert (df["mm_balance_cv_90d"] <= 3).all()


def test_active_days_bounded() -> None:
    df = generate(n=500, seed=42, missingness_rate=0.0)
    assert (df["mm_active_days_90d"] >= 0).all()
    assert (df["mm_active_days_90d"] <= 90).all()


def test_net_flow_is_consistent() -> None:
    df = generate(n=500, seed=42, missingness_rate=0.0)
    recomputed = df["mm_inflow_total_90d"] - df["mm_outflow_total_90d"]
    np.testing.assert_allclose(df["mm_net_flow_90d"].to_numpy(), recomputed.to_numpy())


def test_label_correlates_with_balance_volatility() -> None:
    """Sanity check: distressed/volatile borrowers have higher balance CV and default more."""
    df = generate(n=10_000, seed=42, missingness_rate=0.0)
    high_cv = df[df["mm_balance_cv_90d"] > df["mm_balance_cv_90d"].median()]
    low_cv = df[df["mm_balance_cv_90d"] <= df["mm_balance_cv_90d"].median()]
    assert high_cv[TARGET].mean() > low_cv[TARGET].mean(), (
        "Expected higher default rate among high-balance-volatility borrowers"
    )
