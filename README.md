# agri-credit-score

ML-powered credit scoring for smallholder farmers and agri-SMEs in Tanzania, built on alternative data (mobile money behavior, agroclimatic context, borrower attributes).

**Status:** v0 pre-pilot. Synthetic data only. Not production.

**Research context:** This project accompanies a working paper exploring alternative-data credit scoring methodology for East African microfinance institutions. See `docs/SPEC.md` for the full technical specification and `docs/WORKING_PAPER.md` for the accompanying research writeup.

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/charlestonjm21/agri-credit-score.git
cd agri-credit-score
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Generate synthetic data
python -m agri_credit_score.synthetic.generator --n 10000 --out data/synthetic/borrowers.parquet

# Train model
python -m agri_credit_score.models.train --data data/synthetic/borrowers.parquet --out models/

# Run evaluation
python -m agri_credit_score.models.evaluate --model models/xgb.pkl --data data/synthetic/borrowers.parquet

# Start API
uvicorn agri_credit_score.api.main:app --reload

# Start dashboard (separate terminal)
streamlit run dashboard/app.py
```

Or use the Makefile:

```bash
make install
make data
make train
make serve     # starts API
make dashboard # starts Streamlit
make test
```

---

## Why this exists

Tanzanian smallholder farmers and agri-SMEs are systematically excluded from formal credit because traditional lenders lack the data to price their risk. Mobile money penetration is high (M-Pesa, Tigo Pesa, Airtel Money), weather and commodity price data is publicly available, and SACCOs and MFIs have the borrower relationships. What's missing is the scoring infrastructure that connects these signals to credit decisions.

This project builds that infrastructure as an open, reproducible reference implementation. It is not a lender. It is not a product. It is a methodology and a technical artifact that lenders can inspect, critique, and pilot on their own historical data.

---

## What's in the box

- A synthetic data generator calibrated to published Tanzanian financial inclusion base rates
- A feature pipeline covering mobile money behavioral signals, agroclimatic context, and borrower attributes
- An XGBoost scoring model with isotonic calibration and SHAP-based explanations
- A logistic regression baseline for interpretability comparison
- A FastAPI service exposing `/v1/score` with structured explainability in every response
- A Streamlit dashboard for single-borrower scoring and batch analysis demos
- A fairness audit reporting AUC and FPR disparities across gender and region

---

## Swahili summary

Mradi huu unaunda mfumo wa kutathmini uwezo wa wakulima wadogo na biashara ndogo za kilimo kupata mikopo, kwa kutumia data mbadala kama historia ya miamala ya simu, hali ya hewa, na bei za mazao. Ni mfumo wazi wa utafiti unaolenga kusaidia taasisi za fedha Tanzania (SACCOs na MFIs) kuboresha maamuzi yao ya mikopo kwa wateja ambao hawana rekodi rasmi za benki.

---

## Citing this work

If you use this code or methodology in research or pilot work, please cite:

> Mayenga, C. (2026). Alternative-data credit scoring for smallholder agriculture in Tanzania: A reproducible reference implementation. Working paper.

---

## License

MIT. See LICENSE.

## Contact

Charlestone Mayenga — North Carolina A&T State University
