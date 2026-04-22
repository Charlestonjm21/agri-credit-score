# CLAUDE.md

This file gives Claude Code context for working on this project. Read it first.

## Project: agri-credit-score

ML-powered credit scoring for smallholder farmers and agri-SMEs in Tanzania, using alternative data: mobile money behavior, agroclimatic context, and borrower attributes. Pre-pilot research prototype with synthetic data only. Not production.

Author: Charlestone Mayenga (NC A&T, Class of 2027). This is a side project built 5–10 hours per week, targeting a retrospective backtest with a Tanzanian lender by Day 90. See `docs/SPEC.md` for the full technical specification and `README.md` for the user-facing overview.

## Stack

- Python 3.10+
- Package layout: `src/agri_credit_score/` (PEP 517 src layout, installed editable)
- Data: pandas, parquet, numpy
- ML: XGBoost (primary), scikit-learn (LR baseline, calibration, preprocessing), Optuna (hyperparameter search), SHAP (explainability)
- API: FastAPI with Pydantic v2 schemas, uvicorn
- Dashboard: Streamlit + Plotly
- Tests: pytest + httpx TestClient
- Lint: ruff (configured in `pyproject.toml`, N803/N806 intentionally ignored — ML convention uses uppercase X)

## Common commands

```
make install    # editable install with dev deps
make data       # generate synthetic borrowers
make train      # train XGBoost + LR baseline
make evaluate   # run full eval with fairness audit
make serve      # FastAPI on :8000
make dashboard  # Streamlit on :8501
make test       # pytest with coverage
make lint       # ruff check
```

Install uses `pip install -e ".[dev]" --break-system-packages` on system Python; on WSL/Ubuntu prefer a venv: `python -m venv .venv && source .venv/bin/activate` before installing.

## Conventions Claude should follow

- **Feature lists are single-source-of-truth in `src/agri_credit_score/config.py`.** If you add a feature, update the lists there and the Pydantic schemas in `api/schemas.py` in the same commit. The synthetic generator, preprocessing pipeline, API, and dashboard all pull from those lists.
- **XGBoost handles categoricals natively** via `enable_categorical=True` and `.astype("category")` on input DataFrames. Do NOT one-hot encode before passing to XGBoost. The LR baseline is the only path that uses the preprocessing pipeline.
- **Calibration uses sklearn 1.6+'s `FrozenEstimator` + `CalibratedClassifierCV(method="isotonic")`.** The old `cv="prefit"` API was removed in 1.8. Do not revert.
- **Explainer unwraps through `calibrated_classifiers_[0].estimator.estimator`** to reach the raw XGB. See `models/explain.py`.
- **No PII in logs.** The API hashes borrower IDs before logging. Do not log request bodies.
- **Synthetic data has a deliberate AUC ceiling** via `MAX_ACHIEVABLE_AUC_NOISE` in `synthetic/generator.py`. If you see AUC > 0.85 on synthetic data, the ceiling has been bypassed and something is wrong. Target range: 0.75–0.82.
- **Every prediction must return top-5 SHAP attributions with human-readable labels.** Labels come from `FEATURE_LABELS` in `config.py`. A score without an explanation is unshippable.
- **Keep `docs/SPEC.md` authoritative.** When making design changes, update SPEC.md in the same PR and reference it in the commit message.

## What NOT to do

- Don't add real borrower data, real mobile money transaction data, or real PII to this repo. Ever. Tanzania's Personal Data Protection Act (2022) applies once real data is involved; v0 is synthetic-only by design.
- Don't introduce cloud dependencies (S3, GCS, SageMaker, etc.) in v0. Everything must run on a laptop.
- Don't add a mobile app, lender dashboard, or farmer-facing UI to v0 scope. The deliverables are: model, API, demo dashboard, working paper. Anything else is scope creep.
- Don't swap XGBoost for a deep learning model. For tabular credit data at this sample size, gradient boosting is the right choice and is what lenders will trust.
- Don't remove the logistic regression baseline. It exists as an interpretability anchor and a sanity check. If XGBoost ever fails to beat it, that's a signal, not noise.

## Repo structure

```
agri-credit-score/
├── docs/
│   ├── SPEC.md                    # authoritative technical spec
│   └── WORKING_PAPER.md           # TODO: draft by week 8
├── src/agri_credit_score/
│   ├── config.py                  # feature lists, labels, risk bands
│   ├── features/pipeline.py       # sklearn preprocessor (LR baseline only)
│   ├── synthetic/
│   │   ├── archetypes.py          # 4 borrower archetypes
│   │   └── generator.py           # CLI: python -m ...generator
│   ├── models/
│   │   ├── train.py               # Optuna + XGBoost + calibration + LR baseline
│   │   ├── evaluate.py            # metrics + fairness audit
│   │   └── explain.py             # SHAP wrapper
│   └── api/
│       ├── schemas.py             # Pydantic request/response
│       └── main.py                # FastAPI app with lifespan
├── dashboard/app.py               # Streamlit demo
├── tests/                         # pytest suite
└── data/synthetic/                # gitignored, reproduce from seed 42
```

## Current status

- v0 scaffold complete
- Synthetic generator validated (default rate ~22%, plausible)
- XGBoost trained on 5000 synthetic rows: ROC-AUC 0.78, PR-AUC 0.51, Brier 0.14, ECE 0.04
- LR baseline: ROC-AUC 0.77
- 22/22 tests pass
- API smoke-tested via TestClient
- Streamlit dashboard: 3 tabs (single borrower, batch, model info)

## Next priorities

1. Draft `docs/WORKING_PAPER.md` — 8–12 page technical report for arXiv/SSRN
2. Create outreach tracker (separate from this repo) — 50 SACCO/MFI contacts to interview
3. Add time-based holdout to training (train on months 1–18, test on months 19–24) to simulate retrospective backtest
4. Swahili README translation review (current version is a placeholder)

## Context about the founder

Charlestone is a CS junior at North Carolina A&T, heading to University of Michigan for SROP this summer under Dr. David Jurgens (computational social science / NLP). This project is personal — he has family and business ties in Arusha (Wakanda Land Limited) and wants to build financial infrastructure for East Africa long-term. The side-project nature of this build is a constraint, not a lack of ambition.
