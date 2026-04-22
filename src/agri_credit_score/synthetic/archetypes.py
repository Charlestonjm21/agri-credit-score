"""Borrower archetype definitions for synthetic data generation.

Four archetypes are defined, each with:
- A population share
- A baseline default propensity
- Feature distribution parameters (means and standard deviations)

These distributions are rough approximations informed by published
financial inclusion research in Tanzania. They are NOT fitted to
real data. The goal is to produce plausible correlations between
alternative-data signals and default outcomes, so that the pipeline
and methodology can be validated end to end.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Archetype:
    name: str
    population_share: float
    base_default_rate: float
    # Mobile money feature parameters (mean, std)
    mm_tx_count_90d: tuple[float, float]
    mm_inflow_total_90d: tuple[float, float]
    mm_outflow_total_90d: tuple[float, float]
    mm_balance_mean_90d: tuple[float, float]
    mm_balance_cv_90d: tuple[float, float]
    mm_unique_counterparties_90d: tuple[float, float]
    mm_recurrence_score: tuple[float, float]
    mm_night_tx_ratio: tuple[float, float]
    mm_weekend_tx_ratio: tuple[float, float]
    mm_days_since_last_tx: tuple[float, float]
    mm_active_days_90d: tuple[float, float]
    # Agri context
    rainfall_anomaly_90d: tuple[float, float]
    ndvi_current: tuple[float, float]
    # Borrower
    years_farming: tuple[float, float]
    prior_loans_count: tuple[float, float]
    # Regions skewed toward for this archetype
    regions: list[str] = field(default_factory=list)


ARCHETYPES: list[Archetype] = [
    Archetype(
        name="stable_smallholder",
        population_share=0.55,
        base_default_rate=0.05,
        mm_tx_count_90d=(140, 35),
        mm_inflow_total_90d=(2_400_000, 600_000),
        mm_outflow_total_90d=(2_300_000, 580_000),
        mm_balance_mean_90d=(55_000, 20_000),
        mm_balance_cv_90d=(0.45, 0.15),
        mm_unique_counterparties_90d=(20, 6),
        mm_recurrence_score=(0.42, 0.12),
        mm_night_tx_ratio=(0.06, 0.03),
        mm_weekend_tx_ratio=(0.22, 0.06),
        mm_days_since_last_tx=(2, 2),
        mm_active_days_90d=(68, 10),
        rainfall_anomaly_90d=(0.05, 0.4),
        ndvi_current=(0.62, 0.1),
        years_farming=(14, 7),
        prior_loans_count=(2.5, 1.5),
        regions=["TZ-01", "TZ-02", "TZ-18"],  # Arusha, Dar, Kilimanjaro
    ),
    Archetype(
        name="volatile_farmer",
        population_share=0.25,
        base_default_rate=0.18,
        mm_tx_count_90d=(90, 40),
        mm_inflow_total_90d=(1_200_000, 700_000),
        mm_outflow_total_90d=(1_250_000, 720_000),
        mm_balance_mean_90d=(22_000, 18_000),
        mm_balance_cv_90d=(0.85, 0.25),
        mm_unique_counterparties_90d=(12, 5),
        mm_recurrence_score=(0.22, 0.12),
        mm_night_tx_ratio=(0.12, 0.05),
        mm_weekend_tx_ratio=(0.28, 0.08),
        mm_days_since_last_tx=(5, 4),
        mm_active_days_90d=(45, 12),
        rainfall_anomaly_90d=(-0.35, 0.5),
        ndvi_current=(0.48, 0.12),
        years_farming=(10, 8),
        prior_loans_count=(1.2, 1.3),
        regions=["TZ-03", "TZ-04", "TZ-05"],  # Dodoma, Singida, etc.
    ),
    Archetype(
        name="new_entrant",
        population_share=0.15,
        base_default_rate=0.22,
        mm_tx_count_90d=(60, 25),
        mm_inflow_total_90d=(800_000, 400_000),
        mm_outflow_total_90d=(780_000, 390_000),
        mm_balance_mean_90d=(15_000, 10_000),
        mm_balance_cv_90d=(0.95, 0.3),
        mm_unique_counterparties_90d=(8, 4),
        mm_recurrence_score=(0.15, 0.1),
        mm_night_tx_ratio=(0.1, 0.05),
        mm_weekend_tx_ratio=(0.25, 0.08),
        mm_days_since_last_tx=(6, 5),
        mm_active_days_90d=(32, 10),
        rainfall_anomaly_90d=(-0.1, 0.4),
        ndvi_current=(0.52, 0.12),
        years_farming=(3, 2),
        prior_loans_count=(0.3, 0.6),
        regions=["TZ-01", "TZ-02", "TZ-06"],
    ),
    Archetype(
        name="distressed",
        population_share=0.05,
        base_default_rate=0.55,
        mm_tx_count_90d=(40, 25),
        mm_inflow_total_90d=(400_000, 300_000),
        mm_outflow_total_90d=(520_000, 350_000),
        mm_balance_mean_90d=(6_000, 5_000),
        mm_balance_cv_90d=(1.3, 0.4),
        mm_unique_counterparties_90d=(5, 3),
        mm_recurrence_score=(0.08, 0.08),
        mm_night_tx_ratio=(0.22, 0.1),
        mm_weekend_tx_ratio=(0.32, 0.1),
        mm_days_since_last_tx=(14, 12),
        mm_active_days_90d=(18, 10),
        rainfall_anomaly_90d=(-0.7, 0.5),
        ndvi_current=(0.35, 0.12),
        years_farming=(8, 7),
        prior_loans_count=(1.0, 1.5),
        regions=["TZ-07", "TZ-08"],
    ),
]

CROP_BY_REGION = {
    "TZ-01": ["maize", "coffee", "horticulture"],
    "TZ-02": ["rice", "horticulture"],
    "TZ-03": ["maize", "beans"],
    "TZ-04": ["maize", "beans", "cotton"],
    "TZ-05": ["cotton", "maize"],
    "TZ-06": ["maize", "coffee"],
    "TZ-07": ["maize", "beans"],
    "TZ-08": ["cotton", "rice"],
    "TZ-18": ["coffee", "maize"],
}

ALL_REGIONS = list(CROP_BY_REGION.keys())
