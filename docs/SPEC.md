# agri-credit-score: v0 Technical Specification

**Author:** Charlestone Mayenga
**Version:** 0.1.0 (side-project build, 90-day scope)
**Status:** Pre-pilot. Synthetic data only.

---

## 1. Purpose and scope

This document specifies v0 of an ML-powered credit scoring system for smallholder farmers and agri-SMEs in Tanzania. The system ingests alternative data — mobile money transaction patterns, agroclimatic context, and basic borrower attributes — and returns a calibrated probability of default along with explainability artifacts suitable for a loan-committee review.

**What v0 is:** a reproducible Python pipeline, a FastAPI scoring service, a Streamlit demo dashboard, a synthetic data generator, and a working paper describing the methodology.

**What v0 is not:** a production system, a product with real customers, or a model trained on real borrower data. The goal of v0 is to validate the methodology rigorously enough that a Tanzanian lender will agree to a retrospective backtest on their historical loan book.

---

## 2. Design principles

**Explainability over accuracy.** A SACCO loan committee must be able to read the top features contributing to any score. SHAP values are a required output of every prediction, not a nice-to-have.

**Calibration over discrimination.** AUC is the headline metric, but predicted probabilities must be well-calibrated (Brier score and reliability diagrams). Lenders set thresholds on probabilities, not ranks.

**Offline-first evaluation.** Every feature, model, and evaluation protocol must be reproducible from a seed. No cloud dependencies for v0.

**Honest synthetic data.** The synthetic data generator must reflect published base rates and feature distributions from Tanzanian financial inclusion research. Overly clean synthetic data that produces 0.95 AUC is worse than useless — it creates false confidence.

**API-first packaging.** The scoring logic ships behind a stable REST interface so the same model can serve a Streamlit dashboard, a partner integration, and future mobile clients without rewrites.

---

## 3. System architecture

```
┌─────────────────────┐     ┌──────────────────────┐
│ Synthetic data      │────▶│ Feature engineering  │
│ generator           │     │ pipeline             │
└─────────────────────┘     └──────────┬───────────┘
                                       │
                                       ▼
                            ┌──────────────────────┐
                            │ Model training       │
                            │ (XGBoost + LR base)  │
                            └──────────┬───────────┘
                                       │
                                       ▼
                            ┌──────────────────────┐
                            │ Persisted artifacts  │
                            │ (model, SHAP expl.)  │
                            └──────────┬───────────┘
                                       │
                     ┌─────────────────┼──────────────────┐
                     ▼                 ▼                  ▼
            ┌────────────────┐ ┌───────────────┐ ┌───────────────┐
            │ FastAPI        │ │ Streamlit     │ │ Batch scoring │
            │ /score endpoint│ │ dashboard     │ │ CLI           │
            └────────────────┘ └───────────────┘ └───────────────┘
```

---

## 4. Feature schema

Three feature families, 28 features total in v0. All feature names are snake_case and documented in `src/agri_credit_score/features/schema.py`.

### 4.1 Mobile money behavioral features (12)

Derived from a 90-day rolling window of transaction history.

| Feature | Type | Description |
|---|---|---|
| `mm_tx_count_90d` | int | Total transactions in last 90 days |
| `mm_inflow_total_90d` | float | TZS received |
| `mm_outflow_total_90d` | float | TZS sent |
| `mm_net_flow_90d` | float | Inflow minus outflow |
| `mm_balance_mean_90d` | float | Mean end-of-day balance |
| `mm_balance_cv_90d` | float | Coefficient of variation of daily balance (volatility proxy) |
| `mm_unique_counterparties_90d` | int | Distinct phone numbers transacted with |
| `mm_recurrence_score` | float | Fraction of transactions matching recurring patterns (rent, agri-input suppliers) |
| `mm_night_tx_ratio` | float | Fraction of transactions between 22:00 and 05:00 |
| `mm_weekend_tx_ratio` | float | Fraction of transactions on Saturday/Sunday |
| `mm_days_since_last_tx` | int | Recency |
| `mm_active_days_90d` | int | Days with at least one transaction |

### 4.2 Agri-contextual features (10)

Derived from region (GPS or ward-level admin code), primary crop, and seasonal calendar.

| Feature | Type | Description |
|---|---|---|
| `region_code` | str | Tanzanian region (categorical, one-hot encoded) |
| `primary_crop` | str | Maize, coffee, beans, rice, cotton, horticulture, other |
| `rainfall_anomaly_30d` | float | Deviation from 10-year mean, CHIRPS data |
| `rainfall_anomaly_90d` | float | Same, 90-day window |
| `ndvi_current` | float | Vegetation index proxy for crop health (MODIS) |
| `ndvi_trend_30d` | float | 30-day slope of NDVI |
| `commodity_price_trend_30d` | float | 30-day slope of regional market price for primary crop |
| `commodity_price_volatility_30d` | float | Price CV over 30 days |
| `days_to_harvest` | int | Estimated days until next harvest (cycle-based) |
| `season_phase` | str | Planting, growing, harvest, off-season (categorical) |

### 4.3 Borrower and loan features (6)

| Feature | Type | Description |
|---|---|---|
| `age` | int | Borrower age, years |
| `gender` | str | M, F, Other |
| `years_farming` | int | Self-reported |
| `loan_amount_tzs` | float | Requested principal |
| `loan_term_days` | int | Requested tenor |
| `prior_loans_count` | int | Count of prior formal loans (zero-inflated) |

**Missing data:** All numeric features support missingness. XGBoost handles natively; logistic regression baseline uses median imputation with a missing-indicator column.

---

## 5. Model specification

### 5.1 Primary model: XGBoost

- **Objective:** `binary:logistic`
- **Eval metric:** `aucpr` (precision-recall AUC; robust to class imbalance which is expected in loan default data)
- **Hyperparameter search:** Bayesian optimization via Optuna, 50 trials, 5-fold stratified CV
- **Search space:** max_depth [3, 10], learning_rate [0.01, 0.3], n_estimators [100, 1000] with early stopping, subsample [0.6, 1.0], colsample_bytree [0.6, 1.0], reg_alpha [0, 1], reg_lambda [0, 1]
- **Class weights:** `scale_pos_weight` set to ratio of negatives to positives in training fold
- **Calibration:** Isotonic regression on a held-out calibration set (10% of training data)

### 5.2 Baseline: Logistic regression

Required as an interpretability anchor and sanity check. If XGBoost doesn't materially outperform LR on AUC and Brier score, something is wrong with the features.

- **Preprocessing:** Median imputation + missing indicators, standard scaling, one-hot encoding for categoricals
- **Regularization:** L2, C tuned via 5-fold CV

### 5.3 Explainability

Every prediction returned by the API must include:

- The raw probability
- A risk band (Low / Moderate / Elevated / High) derived from calibrated quartiles of the training distribution
- Top 5 features by absolute SHAP value, each with: feature name, human-readable label, contribution direction (increases/decreases risk), and magnitude

---

## 6. Evaluation protocol

### 6.1 Metrics

**Primary:**
- ROC-AUC (discrimination)
- PR-AUC (performance under imbalance)
- Brier score (calibration)
- Expected Calibration Error (ECE) with 10 bins

**Secondary:**
- KS statistic (regulatory standard in credit)
- Lift at top decile
- Reliability diagram (visual)

### 6.2 Validation strategy

- **5-fold stratified cross-validation** on the full synthetic dataset
- **Time-based holdout:** Use `spread_months=24` in the synthetic generator to distribute borrowers across 24 monthly cohorts (linearly recency-weighted). The CLI flag `--split-mode time` then applies the following protocol:
  1. Sort the dataset by `as_of_date`. The first 80% of rows (by date) form the **pool**; the last 20% are the **true future holdout** (unseen during all training).
  2. Within the pool, two evaluations are run side-by-side to isolate the effect of split strategy while holding data volume and archetype distribution constant:
     - **`time_split_metrics`**: train on the pool's first 80% (earliest cohorts), evaluate on the pool's last 20% (most recent cohorts).
     - **`random_split_metrics`**: random 80/20 split of the same pool; train on random 80%, evaluate on random 20%.
  3. The primary model artifact saved is the time-split-trained model. It is also evaluated on the true future holdout and reported as **`future_holdout_metrics`**.
  - The gap between `random_split_metrics` and `time_split_metrics` measures within-pool temporal drift. The gap between `time_split_metrics` and `future_holdout_metrics` measures out-of-pool degradation.
- **Fairness audit:** AUC and FPR parity across gender and across the three main regional groupings (Northern, Coastal, Southern Highlands). Report disparities even if small.

### 6.3 Target performance on synthetic data

Published alternative-data credit models in East Africa report AUCs of 0.72–0.80 on real data. On well-constructed synthetic data that includes realistic noise, aim for **AUC 0.75–0.82**. Higher suggests the synthetic data is too clean; lower suggests a feature engineering bug.

---

## 7. API specification

### 7.1 Endpoints

**POST /v1/score**

Request body (JSON):
```json
{
  "borrower_id": "string",
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
    "active_days_90d": 67
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
    "season_phase": "growing"
  },
  "borrower": {
    "age": 34,
    "gender": "F",
    "years_farming": 12,
    "prior_loans_count": 2
  },
  "loan": {
    "loan_amount_tzs": 500000,
    "loan_term_days": 180
  }
}
```

Response body (JSON):
```json
{
  "borrower_id": "string",
  "model_version": "0.1.0",
  "scored_at": "2026-04-22T14:00:00Z",
  "default_probability": 0.0874,
  "risk_band": "Low",
  "top_features": [
    {
      "name": "mm_active_days_90d",
      "label": "Active days on mobile money (last 90)",
      "direction": "decreases_risk",
      "shap_value": -0.412
    }
  ],
  "calibration_notes": "Probabilities calibrated via isotonic regression on holdout set."
}
```

**GET /v1/health** — liveness check, returns model version and load timestamp.

**GET /v1/model-info** — returns model metadata, training date, feature list, performance on holdout set.

### 7.2 Non-functional requirements

- **Latency:** p95 < 200ms on a single score (model is small; this is trivial)
- **Validation:** Pydantic models reject malformed requests with 422 errors
- **Logging:** Structured JSON logs, no PII written to logs
- **Auth:** Not in v0. Add API key auth before any external demo.

---

## 8. Synthetic data generation

The synthetic data generator lives in `src/agri_credit_score/synthetic/` and produces a dataset of N borrowers with labeled default outcomes. Design choices:

**Base default rate:** 12%, calibrated to published smallholder agricultural loan default rates in Tanzania (8–15% range per FSD reports). Configurable.

**Borrower archetypes:** Four archetypes with different feature distributions and default propensities:
- *Stable smallholder* (55% of population, 5% default rate): consistent mobile money activity, diversified counterparties, stable rainfall region
- *Volatile farmer* (25%, 18% default): irregular mobile money, high balance CV, drought-exposed region
- *New entrant* (15%, 22% default): low prior loan count, short mobile money history, thin file
- *Distressed* (5%, 55% default): declining balance trend, high night-tx ratio (proxy for stress), negative rainfall anomaly

**Realistic noise:** Features are drawn from archetype-conditional distributions, then perturbed with Gaussian noise (feature-specific sigma) and 5% missingness injected uniformly at random.

**Signal ceiling:** The generator intentionally caps the maximum achievable AUC around 0.85 by introducing irreducible noise in the label-generating process. Models that exceed this are overfitting to the generator.

**Reproducibility:** Every generation run takes a seed. Default seed 42.

---

## 9. Repo structure

```
agri-credit-score/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── Makefile
├── docs/
│   ├── SPEC.md                 # this file
│   ├── WORKING_PAPER.md        # draft working paper (add week 8)
│   └── FEATURE_DICTIONARY.md
├── src/agri_credit_score/
│   ├── __init__.py
│   ├── config.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── schema.py           # Pydantic models, feature dict
│   │   └── pipeline.py         # sklearn ColumnTransformer
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py            # XGBoost + LR training
│   │   ├── calibrate.py        # isotonic calibration
│   │   ├── evaluate.py         # metrics, fairness audit
│   │   └── explain.py          # SHAP wrapper
│   ├── synthetic/
│   │   ├── __init__.py
│   │   ├── archetypes.py       # borrower archetype definitions
│   │   └── generator.py        # main generation function
│   └── api/
│       ├── __init__.py
│       ├── main.py             # FastAPI app
│       └── schemas.py          # request/response Pydantic
├── dashboard/
│   └── app.py                  # Streamlit demo
├── tests/
│   ├── test_synthetic.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
├── notebooks/
│   └── 01_eda.ipynb
└── data/
    ├── synthetic/              # generated datasets (gitignored)
    └── external/               # reference data (crop calendars, etc.)
```

---

## 10. 90-day delivery plan

| Week | Milestone |
|---|---|
| 1 | Repo skeleton, CI, pyproject.toml, initial README |
| 2 | Synthetic data generator v1, smoke-tested to produce plausible distributions |
| 3 | Feature pipeline, logistic regression baseline, 5-fold CV report |
| 4 | XGBoost training with Optuna hyperparameter search, calibration |
| 5 | SHAP integration, fairness audit across gender and region |
| 6 | FastAPI service with scoring endpoint, Dockerfile |
| 7 | Streamlit dashboard with single-borrower and batch modes |
| 8 | Working paper draft v1 on arXiv/SSRN |
| 9 | Outreach to 3 most engaged Phase 1 interviewees with paper + repo |
| 10–13 | Iterate based on feedback; target one retrospective pilot commitment |

---

## 11. Risks and mitigations

**Risk: Synthetic data is too clean, model performs unrealistically.**
Mitigation: Signal ceiling in generator; compare distributions to any published real-data moments available in the literature.

**Risk: Feature engineering introduces label leakage.**
Mitigation: All features are computed strictly as-of `as_of_date`, not using any information after that point. Unit tests enforce this.

**Risk: Regulatory exposure when moving to real data.**
Mitigation: Tanzania Personal Data Protection Act, 2022 review before any real data touches the system. Document data handling, retention, and consent patterns now even though no real data is used.

**Risk: Model performance on real data is materially worse than on synthetic data.**
Mitigation: This is the point of the retrospective backtest. Expect degradation; frame it honestly in conversations with lenders.

---

## 12. Definition of done for v0

- [ ] All code passes `pytest` and `ruff`
- [ ] Synthetic data generator produces 10,000 borrowers in under 30 seconds
- [ ] XGBoost model achieves AUC ≥ 0.75 on time-based holdout
- [ ] Calibration ECE ≤ 0.05 on holdout
- [ ] Fairness audit completed, disparities documented
- [ ] FastAPI service passes integration tests
- [ ] Streamlit dashboard runs locally with `make dashboard`
- [ ] Working paper drafted, on preprint server
- [ ] GitHub repo public, README in English and Swahili
- [ ] At least 3 stakeholder interviews cite the repo or paper
