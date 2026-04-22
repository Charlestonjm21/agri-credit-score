"""Central configuration constants for the scoring system."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_VERSION = "0.1.0"

# Feature lists live here so train/serve/eval stay in sync.
MOBILE_MONEY_FEATURES = [
    "mm_tx_count_90d",
    "mm_inflow_total_90d",
    "mm_outflow_total_90d",
    "mm_net_flow_90d",
    "mm_balance_mean_90d",
    "mm_balance_cv_90d",
    "mm_unique_counterparties_90d",
    "mm_recurrence_score",
    "mm_night_tx_ratio",
    "mm_weekend_tx_ratio",
    "mm_days_since_last_tx",
    "mm_active_days_90d",
]

AGRI_NUMERIC_FEATURES = [
    "rainfall_anomaly_30d",
    "rainfall_anomaly_90d",
    "ndvi_current",
    "ndvi_trend_30d",
    "commodity_price_trend_30d",
    "commodity_price_volatility_30d",
    "days_to_harvest",
]

AGRI_CATEGORICAL_FEATURES = ["region_code", "primary_crop", "season_phase"]

BORROWER_NUMERIC_FEATURES = ["age", "years_farming", "prior_loans_count"]
BORROWER_CATEGORICAL_FEATURES = ["gender"]

LOAN_FEATURES = ["loan_amount_tzs", "loan_term_days"]

NUMERIC_FEATURES = (
    MOBILE_MONEY_FEATURES
    + AGRI_NUMERIC_FEATURES
    + BORROWER_NUMERIC_FEATURES
    + LOAN_FEATURES
)
CATEGORICAL_FEATURES = AGRI_CATEGORICAL_FEATURES + BORROWER_CATEGORICAL_FEATURES
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "defaulted"

# Human-readable labels for explanations.
FEATURE_LABELS = {
    "mm_tx_count_90d": "Mobile money transactions (last 90 days)",
    "mm_inflow_total_90d": "Total money received (last 90 days)",
    "mm_outflow_total_90d": "Total money sent (last 90 days)",
    "mm_net_flow_90d": "Net cash flow (last 90 days)",
    "mm_balance_mean_90d": "Average mobile money balance",
    "mm_balance_cv_90d": "Balance volatility",
    "mm_unique_counterparties_90d": "Unique transaction counterparties",
    "mm_recurrence_score": "Recurring payment pattern strength",
    "mm_night_tx_ratio": "Fraction of late-night transactions",
    "mm_weekend_tx_ratio": "Fraction of weekend transactions",
    "mm_days_since_last_tx": "Days since last transaction",
    "mm_active_days_90d": "Active mobile money days (last 90)",
    "rainfall_anomaly_30d": "Rainfall deviation from normal (30d)",
    "rainfall_anomaly_90d": "Rainfall deviation from normal (90d)",
    "ndvi_current": "Current crop vegetation index",
    "ndvi_trend_30d": "Vegetation trend (last 30 days)",
    "commodity_price_trend_30d": "Local crop price trend",
    "commodity_price_volatility_30d": "Local crop price volatility",
    "days_to_harvest": "Days until next harvest",
    "age": "Borrower age",
    "years_farming": "Years farming experience",
    "prior_loans_count": "Number of prior loans",
    "loan_amount_tzs": "Requested loan amount (TZS)",
    "loan_term_days": "Requested loan term (days)",
    "region_code": "Tanzanian region",
    "primary_crop": "Primary crop",
    "season_phase": "Current season phase",
    "gender": "Gender",
}

# Risk band thresholds on calibrated default probability.
RISK_BANDS = [
    (0.00, 0.10, "Low"),
    (0.10, 0.25, "Moderate"),
    (0.25, 0.50, "Elevated"),
    (0.50, 1.01, "High"),
]


def risk_band(probability: float) -> str:
    """Map a probability to a human-readable risk band."""
    for lo, hi, label in RISK_BANDS:
        if lo <= probability < hi:
            return label
    return "High"
