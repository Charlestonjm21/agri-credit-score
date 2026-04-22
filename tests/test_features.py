"""Tests for the preprocessing pipeline."""
from __future__ import annotations

import numpy as np

from agri_credit_score.config import ALL_FEATURES
from agri_credit_score.features.pipeline import build_preprocessor
from agri_credit_score.synthetic.generator import generate


def test_preprocessor_fits_and_transforms() -> None:
    df = generate(n=300, seed=42)
    X = df[ALL_FEATURES]
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(df)
    assert Xt.shape[1] > len(ALL_FEATURES), (
        "One-hot encoding should expand categorical columns"
    )
    assert not np.isnan(Xt).any(), "Imputation should eliminate NaNs"


def test_preprocessor_handles_unseen_categories() -> None:
    df_train = generate(n=500, seed=42)
    df_test = generate(n=100, seed=99)
    # Inject an unseen region into the test set
    df_test.loc[0, "region_code"] = "TZ-XX-UNSEEN"

    pre = build_preprocessor()
    pre.fit(df_train[ALL_FEATURES])
    Xt = pre.transform(df_test[ALL_FEATURES])
    assert Xt.shape[0] == len(df_test)
    assert not np.isnan(Xt).any()
