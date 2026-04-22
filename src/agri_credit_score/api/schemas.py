"""Pydantic request/response schemas for the scoring API."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class MobileMoneyFeatures(BaseModel):
    tx_count_90d: int = Field(..., ge=0)
    inflow_total_90d: float = Field(..., ge=0)
    outflow_total_90d: float = Field(..., ge=0)
    balance_mean_90d: float = Field(..., ge=0)
    balance_cv_90d: float = Field(..., ge=0)
    unique_counterparties_90d: int = Field(..., ge=0)
    recurrence_score: float = Field(..., ge=0, le=1)
    night_tx_ratio: float = Field(..., ge=0, le=1)
    weekend_tx_ratio: float = Field(..., ge=0, le=1)
    days_since_last_tx: int = Field(..., ge=0)
    active_days_90d: int = Field(..., ge=0, le=90)


class AgriContext(BaseModel):
    region_code: str
    primary_crop: Literal[
        "maize", "coffee", "beans", "rice", "cotton", "horticulture", "other"
    ]
    rainfall_anomaly_30d: float
    rainfall_anomaly_90d: float
    ndvi_current: float = Field(..., ge=0, le=1)
    ndvi_trend_30d: float
    commodity_price_trend_30d: float
    commodity_price_volatility_30d: float = Field(..., ge=0)
    days_to_harvest: int = Field(..., ge=0, le=365)
    season_phase: Literal["planting", "growing", "harvest", "off_season"]


class BorrowerFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100)
    gender: Literal["F", "M", "Other"]
    years_farming: int = Field(..., ge=0)
    prior_loans_count: int = Field(..., ge=0)


class LoanFeatures(BaseModel):
    loan_amount_tzs: float = Field(..., gt=0)
    loan_term_days: int = Field(..., gt=0, le=730)


class ScoreRequest(BaseModel):
    borrower_id: str = Field(..., min_length=1, max_length=64)
    as_of_date: date
    mobile_money: MobileMoneyFeatures
    agri_context: AgriContext
    borrower: BorrowerFeatures
    loan: LoanFeatures

    @field_validator("borrower_id")
    @classmethod
    def _no_pii_in_id(cls, v: str) -> str:
        # Cheap guard: reject obvious phone-number-shaped IDs
        digits_only = "".join(ch for ch in v if ch.isdigit())
        if len(digits_only) >= 9 and digits_only == v.replace("+", "").replace(" ", ""):
            raise ValueError(
                "borrower_id looks like a phone number; use an anonymized identifier"
            )
        return v


class FeatureAttribution(BaseModel):
    name: str
    label: str
    direction: Literal["increases_risk", "decreases_risk"]
    shap_value: float


class ScoreResponse(BaseModel):
    borrower_id: str
    model_version: str
    scored_at: datetime
    default_probability: float = Field(..., ge=0, le=1)
    risk_band: Literal["Low", "Moderate", "Elevated", "High"]
    top_features: list[FeatureAttribution]
    calibration_notes: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str
    model_loaded_at: datetime


class ModelInfoResponse(BaseModel):
    model_version: str
    trained_at: str
    n_train_rows: int
    features: list[str]
    test_metrics: dict
