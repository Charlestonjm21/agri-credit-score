"""Tests for the FastAPI scoring service."""
from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agri_credit_score.models.train import train_xgboost
from agri_credit_score.synthetic.generator import generate


@pytest.fixture(scope="module")
def trained_client(tmp_path_factory: pytest.TempPathFactory):
    """Train a tiny model once, point the app at it, and yield a TestClient."""
    models_dir = tmp_path_factory.mktemp("models")

    df = generate(n=800, seed=42)
    model, _ = train_xgboost(df, n_trials=3, seed=42)
    with open(models_dir / "xgb.pkl", "wb") as f:
        pickle.dump(model, f)

    # Patch the config MODELS_DIR before importing the app
    import agri_credit_score.config as config

    config.MODELS_DIR = Path(models_dir)

    from agri_credit_score.api import main as api_main

    api_main.MODELS_DIR = Path(models_dir)
    api_main.bundle.load()

    client = TestClient(api_main.app)
    return client


def _sample_request() -> dict:
    return {
        "borrower_id": "test-borrower-001",
        "as_of_date": "2026-04-22",
        "mobile_money": {
            "tx_count_90d": 142,
            "inflow_total_90d": 2450000,
            "outflow_total_90d": 2380000,
            "balance_mean_90d": 45000,
            "balance_cv_90d": 0.62,
            "unique_counterparties_90d": 18,
            "recurrence_score": 0.34,
            "night_tx_ratio": 0.08,
            "weekend_tx_ratio": 0.21,
            "days_since_last_tx": 2,
            "active_days_90d": 67,
        },
        "agri_context": {
            "region_code": "TZ-01",
            "primary_crop": "maize",
            "rainfall_anomaly_30d": -0.4,
            "rainfall_anomaly_90d": -0.2,
            "ndvi_current": 0.58,
            "ndvi_trend_30d": 0.02,
            "commodity_price_trend_30d": 0.05,
            "commodity_price_volatility_30d": 0.12,
            "days_to_harvest": 45,
            "season_phase": "growing",
        },
        "borrower": {
            "age": 34,
            "gender": "F",
            "years_farming": 12,
            "prior_loans_count": 2,
        },
        "loan": {
            "loan_amount_tzs": 500000,
            "loan_term_days": 180,
        },
    }


def test_health_returns_ok(trained_client: TestClient) -> None:
    resp = trained_client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_version" in body


def test_score_returns_valid_response(trained_client: TestClient) -> None:
    resp = trained_client.post("/v1/score", json=_sample_request())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert 0 <= body["default_probability"] <= 1
    assert body["risk_band"] in {"Low", "Moderate", "Elevated", "High"}
    assert len(body["top_features"]) == 5
    for feat in body["top_features"]:
        assert feat["direction"] in {"increases_risk", "decreases_risk"}


def test_score_rejects_invalid_payload(trained_client: TestClient) -> None:
    bad = _sample_request()
    bad["mobile_money"]["recurrence_score"] = 2.5  # out of [0, 1]
    resp = trained_client.post("/v1/score", json=bad)
    assert resp.status_code == 422


def test_score_rejects_phone_number_as_id(trained_client: TestClient) -> None:
    bad = _sample_request()
    bad["borrower_id"] = "+255712345678"
    resp = trained_client.post("/v1/score", json=bad)
    assert resp.status_code == 422
