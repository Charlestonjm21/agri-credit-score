"""FastAPI scoring service.

Endpoints
---------
POST /v1/score       Score a single borrower, returns probability + SHAP top-k
GET  /v1/health      Liveness check
GET  /v1/model-info  Model metadata and test-fold metrics

Run locally:
    uvicorn agri_credit_score.api.main:app --reload
"""
from __future__ import annotations

import json
import logging
import pickle
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException

from agri_credit_score.config import (
    CATEGORICAL_FEATURES,
    MODEL_VERSION,
    MODELS_DIR,
    risk_band,
)
from agri_credit_score.models.explain import Explainer

from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    ScoreRequest,
    ScoreResponse,
)

logger = logging.getLogger("agri_credit_score.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


class _ModelBundle:
    """Lazily loaded model + explainer + metadata."""

    def __init__(self) -> None:
        self.model = None
        self.explainer: Explainer | None = None
        self.metadata: dict = {}
        self.loaded_at: datetime | None = None

    def load(self) -> None:
        model_path = MODELS_DIR / "xgb.pkl"
        metadata_path = MODELS_DIR / "metadata.json"
        if not model_path.exists():
            raise RuntimeError(
                f"Model file not found at {model_path}. Run `make train` first."
            )
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        self.explainer = Explainer(self.model)
        self.loaded_at = datetime.now(timezone.utc)
        logger.info("Model loaded from %s", model_path)


bundle = _ModelBundle()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: attempt model load at startup, log but don't crash on failure."""
    try:
        bundle.load()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model load deferred: %s", exc)
    yield


app = FastAPI(
    title="agri-credit-score API",
    version=MODEL_VERSION,
    description="Alternative-data credit scoring for smallholder agriculture in Tanzania.",
    lifespan=lifespan,
)


def _request_to_dataframe(req: ScoreRequest) -> pd.DataFrame:
    """Flatten a request into a single-row DataFrame with correct dtypes."""
    row = {
        # Mobile money
        "mm_tx_count_90d": req.mobile_money.tx_count_90d,
        "mm_inflow_total_90d": req.mobile_money.inflow_total_90d,
        "mm_outflow_total_90d": req.mobile_money.outflow_total_90d,
        "mm_net_flow_90d": (
            req.mobile_money.inflow_total_90d - req.mobile_money.outflow_total_90d
        ),
        "mm_balance_mean_90d": req.mobile_money.balance_mean_90d,
        "mm_balance_cv_90d": req.mobile_money.balance_cv_90d,
        "mm_unique_counterparties_90d": req.mobile_money.unique_counterparties_90d,
        "mm_recurrence_score": req.mobile_money.recurrence_score,
        "mm_night_tx_ratio": req.mobile_money.night_tx_ratio,
        "mm_weekend_tx_ratio": req.mobile_money.weekend_tx_ratio,
        "mm_days_since_last_tx": req.mobile_money.days_since_last_tx,
        "mm_active_days_90d": req.mobile_money.active_days_90d,
        # Agri
        "region_code": req.agri_context.region_code,
        "primary_crop": req.agri_context.primary_crop,
        "rainfall_anomaly_30d": req.agri_context.rainfall_anomaly_30d,
        "rainfall_anomaly_90d": req.agri_context.rainfall_anomaly_90d,
        "ndvi_current": req.agri_context.ndvi_current,
        "ndvi_trend_30d": req.agri_context.ndvi_trend_30d,
        "commodity_price_trend_30d": req.agri_context.commodity_price_trend_30d,
        "commodity_price_volatility_30d": req.agri_context.commodity_price_volatility_30d,
        "days_to_harvest": req.agri_context.days_to_harvest,
        "season_phase": req.agri_context.season_phase,
        # Borrower
        "age": req.borrower.age,
        "gender": req.borrower.gender,
        "years_farming": req.borrower.years_farming,
        "prior_loans_count": req.borrower.prior_loans_count,
        # Loan
        "loan_amount_tzs": req.loan.loan_amount_tzs,
        "loan_term_days": req.loan.loan_term_days,
    }
    from agri_credit_score.config import ALL_FEATURES

    df = pd.DataFrame([row])[ALL_FEATURES]
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df


@app.post("/v1/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    if bundle.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model with `make train` and restart.",
        )

    X = _request_to_dataframe(req)
    prob = float(bundle.model.predict_proba(X)[0, 1])
    band = risk_band(prob)

    assert bundle.explainer is not None
    attributions = bundle.explainer.explain(X, top_k=5)[0]

    # Structured log without PII
    logger.info(
        json.dumps(
            {
                "event": "score",
                "borrower_id_hash": hash(req.borrower_id) & 0xFFFFFFFF,
                "probability": round(prob, 4),
                "risk_band": band,
            }
        )
    )

    return ScoreResponse(
        borrower_id=req.borrower_id,
        model_version=MODEL_VERSION,
        scored_at=datetime.now(timezone.utc),
        default_probability=prob,
        risk_band=band,  # type: ignore[arg-type]
        top_features=attributions,
        calibration_notes="Probabilities calibrated via isotonic regression on holdout set.",
    )


@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if bundle.model is None or bundle.loaded_at is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return HealthResponse(
        status="ok",
        model_version=MODEL_VERSION,
        model_loaded_at=bundle.loaded_at,
    )


@app.get("/v1/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if not bundle.metadata:
        raise HTTPException(status_code=503, detail="Model metadata not available.")
    meta = bundle.metadata
    return ModelInfoResponse(
        model_version=meta.get("model_version", MODEL_VERSION),
        trained_at=meta.get("trained_at", "unknown"),
        n_train_rows=meta.get("n_train_rows", 0),
        features=meta.get("features", []),
        test_metrics=meta.get("xgboost", {}).get("test_metrics", {}),
    )
