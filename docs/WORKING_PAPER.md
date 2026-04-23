# Alternative-Data Credit Scoring for Smallholder Farmers in Tanzania: A Synthetic Benchmark and Retrospective Evaluation Framework

**Charlestone Mayenga**
North Carolina Agricultural and Technical State University

*Working paper — v0.1 draft scaffold. Not for citation. Numerical results pending real-data validation.*

---

## Abstract

<!-- PLACEHOLDER: One paragraph. State the research question (can mobile-money behavior and agroclimatic context reliably signal default risk for smallholder farmers excluded from formal credit bureaus?), summarize the approach (synthetic benchmark with 10,000 borrowers, XGBoost + LR baseline, three-tier evaluation), headline the metrics (ROC-AUC ~0.75 on time-split holdout [metadata.json: xgboost.time_split_metrics.roc_auc]), and close with the honest framing: synthetic results validate methodology, not deployment readiness — the retrospective lender backtest is the central validation step. -->

---

## 1. Introduction

### 1.1 Problem and Motivation

<!-- PLACEHOLDER: Introduce the credit access gap for smallholder farmers and agri-SMEs in sub-Saharan Africa. Note that thin credit files — not creditworthiness — are the primary barrier. [FACT: Tanzania unbanked population per World Bank Findex 2021 or most recent wave] [FACT: share of Tanzanian adults with access to formal credit, Findex] -->

### 1.2 Tanzanian Context

<!-- PLACEHOLDER: Describe the Tanzanian lending landscape: SACCO and MFI dominance in rural credit, mobile money penetration (M-Pesa, Tigopesa, Airtel Money), and the gap between digital financial activity and formal credit access. [FACT: Tanzania mobile money account penetration, GSMA or FinScope Tanzania] [CITE: canonical reference on mobile money adoption in Tanzania or East Africa] -->

### 1.3 Research Question

<!-- PLACEHOLDER: State the question precisely: can a feature schema derived from 90-day mobile-money behavior and agroclimatic signals — with no bureau data — produce well-calibrated default probability estimates suitable for SACCO/MFI underwriting? -->

### 1.4 Contributions

<!-- PLACEHOLDER: Bullet list of concrete contributions:
- An open feature schema (28 features across 3 groups) grounded in published financial-inclusion research
- A behavioral archetype-based synthetic benchmark with known default propensities
- A three-tier evaluation protocol (random split, time split, future holdout) separating methodology validation from temporal generalization
- A SHAP explainability layer that returns top-5 human-readable attributions per prediction
- A deployable FastAPI scoring service + Streamlit dashboard
- An honest framing of what synthetic validation can and cannot establish, and a protocol for the retrospective lender backtest -->

---

## 2. Related Work

### 2.1 Alternative Data for Credit Scoring

<!-- PLACEHOLDER: Summarize the literature on using non-bureau signals (mobile money, psychometric tests, social network data) to score thin-file borrowers. Note the difference between mobile-money volume signals and behavioral pattern signals. [CITE: canonical paper on mobile-money-based credit scoring, e.g. Björkegren & Grissen or similar] [CITE: review of alternative data in emerging-market credit scoring] -->

### 2.2 Digital Finance and Microfinance in East Africa

<!-- PLACEHOLDER: Situate this work relative to M-Shwari, Tala, Branch, and SACCO-specific digital lending in Tanzania. Note the gap: most deployed systems are proprietary and their feature schemas are not published. [CITE: M-Shwari or digital credit in Kenya/Tanzania academic study] [CITE: FinScope Tanzania or FSDT report] -->

### 2.3 Fairness and Bias in Automated Credit Decisions

<!-- PLACEHOLDER: Introduce the fairness audit framing: demographic parity and equalized odds as the two metrics used here, and why gender and region are the protected attributes of concern in the Tanzanian agricultural context. [CITE: Hardt et al. 2016 "Equality of Opportunity in Supervised Learning" or equivalent canonical fairness paper] [CITE: fairness in lending / credit scoring paper specific to developing markets if available] -->

---

## 3. Data

### 3.1 Synthetic Data Design

<!-- PLACEHOLDER: Describe the synthetic generator: 10,000 borrowers [metadata.json: n_total_rows], seed 42 [metadata.json: seed], dataset-level default rate ~22.7% [metadata.json: xgboost.random_split_metrics.test_default_rate = 0.2275]. Explain the AUC ceiling (MAX_ACHIEVABLE_AUC_NOISE in generator.py) that constrains the model to the 0.75–0.82 range on synthetic data, preventing overfitting to a noise-free signal. Note that distributions are informed by published financial-inclusion research but are NOT fitted to real transaction data. [CITE: financial inclusion research that informed the distribution parameters] -->

### 3.2 Borrower Archetypes

<!-- PLACEHOLDER: Describe the four behavioral credit-risk archetypes defined in src/agri_credit_score/synthetic/archetypes.py. Each archetype encodes a distinct pattern of mobile-money behavior and agroclimatic exposure, not a demographic category:

- **stable_smallholder** (55% of population): High transaction frequency, low balance volatility, strong recurrence score, low rainfall stress. Base default rate: 5%.
- **volatile_farmer** (25%): Moderate transaction volume, high balance CV, elevated night-transaction ratio, negative rainfall anomaly. Base default rate: 18%.
- **new_entrant** (15%): Low transaction count, thin prior-loan history, minimal recurrence signal. Base default rate: 22%.
- **distressed** (5%): Very low transaction activity, outflows exceed inflows, extreme balance volatility, severely negative agroclimatic signals. Base default rate: 55%.

[src/agri_credit_score/synthetic/archetypes.py: population_share and base_default_rate fields for each archetype] -->

### 3.3 IID Structure and Temporal Scaffolding

<!-- PLACEHOLDER: Be explicit: the synthetic data is IID by construction. The spread_months parameter distributes borrowers across monthly cohorts with recency-weighted sampling, creating a structural scaffold for time-split analysis, but it does not generate real temporal dynamics (concept drift, seasonal default cycles, macroeconomic shocks). The absence of meaningful drift between time-split and random-split metrics (XGBoost ROC-AUC 0.746 vs. 0.740 [metadata.json: xgboost.time_split_metrics.roc_auc vs. random_split_metrics.roc_auc]) is expected, not a finding. This is noted here so the reader understands why the retrospective lender backtest — not this benchmark — is the methodologically central validation step. -->

---

## 4. Methods

### 4.1 Feature Schema

<!-- PLACEHOLDER: Describe the three feature groups as defined in src/agri_credit_score/config.py:
- **Mobile money behavior** (12 features): transaction counts, inflow/outflow volumes, balance statistics, counterparty diversity, behavioral timing ratios (night, weekend), and activity recency.
- **Agroclimatic context** (10 features, 7 numeric + 3 categorical): rainfall anomalies, NDVI vegetation index and trend, commodity price signals, days-to-harvest, region code, primary crop, and season phase.
- **Borrower attributes and loan terms** (6 features): age, years farming, prior loan count, gender, loan amount (TZS), and loan term (days).
Total: 28 features [metadata.json: features list]. No bureau data, no PII beyond age and gender. -->

### 4.2 Model Family

<!-- PLACEHOLDER: XGBoost (primary model) with enable_categorical=True and native categorical encoding — no one-hot encoding. Logistic regression (baseline) via sklearn preprocessing pipeline. Rationale: gradient boosting is the standard-of-practice for tabular credit data at this sample size; LR provides an interpretability anchor and a sanity check (if XGBoost fails to beat LR, that is a signal). Optuna hyperparameter search with n_trials=50 [TODO: confirm trial count from train.py]. -->

### 4.3 Calibration

<!-- PLACEHOLDER: Post-hoc isotonic calibration using sklearn 1.6+ FrozenEstimator + CalibratedClassifierCV(method="isotonic"). Calibration maps raw XGBoost scores to well-founded default probabilities usable for risk-band assignment. Risk bands are defined in config.py: Low (<10%), Moderate (10–25%), Elevated (25–50%), High (≥50%). -->

### 4.4 SHAP Explainability

<!-- PLACEHOLDER: SHAP TreeExplainer applied to the raw XGBoost model, accessed by unwrapping through the calibration layer (calibrated_classifiers_[0].estimator.estimator — see models/explain.py). Every prediction returns the top-5 feature attributions with human-readable labels from FEATURE_LABELS in config.py. A score without an explanation is not returned. -->

### 4.5 System Architecture

<!-- PLACEHOLDER: Describe the deployable prototype as two services:
- **FastAPI scoring service** (src/agri_credit_score/api/main.py): accepts a structured JSON borrower record, returns calibrated default probability, risk band, and top-5 SHAP attributions. Borrower IDs are hashed before logging; no PII in logs.
- **Streamlit dashboard** (dashboard/app.py): three-tab interface for single-borrower scoring, batch CSV scoring, and model information. Intended for lender demo and methodology review, not production deployment.
Position this as a research prototype running on a laptop, not a cloud deployment. [TODO: note API latency benchmarks if available] -->

---

## 5. Experimental Setup

### 5.1 Three-Tier Evaluation Protocol

<!-- PLACEHOLDER: Define the three evaluation modes and what each validates:
1. **Random split** (80/20, stratified): Baseline i.i.d. performance. Upper bound on what synthetic data can show.
2. **Time split** (80/20, chronological): Tests whether the model degrades when trained on earlier cohorts and evaluated on later ones. On synthetic IID data, results are expected to be similar to random split — and they are.
3. **Future holdout** (20% held out before any training): Simulates a lender deploying the trained model on genuinely unseen borrowers. Most conservative estimate of deployment performance.
[metadata.json: n_pool_rows=8000, n_pool_train_rows=6400, n_future_holdout_rows=2000] -->

### 5.2 Metrics

<!-- PLACEHOLDER: Define the four metrics and their lending-specific relevance:
- **ROC-AUC**: Rank-ordering ability across all thresholds. Threshold-independent.
- **PR-AUC**: Precision-recall tradeoff, more informative than AUC under class imbalance (~22.7% positive rate).
- **Brier score**: Proper scoring rule for calibration quality; measures mean squared error of probability estimates.
- **ECE (Expected Calibration Error)**: Calibration in probability bins; critical for risk-band reliability. [TODO: pull ECE from eval_report.json — run `make evaluate` to generate] -->

### 5.3 Fairness Audit

<!-- PLACEHOLDER: Describe the protected attributes (gender, region_code, archetype) and fairness metrics. Define demographic parity gap (difference in positive prediction rates across groups) and equalized odds gap (difference in TPR and FPR across groups). [CITE: Hardt et al. 2016 or equivalent] [TODO: pull all fairness numbers from eval_report.json — run `make evaluate` to generate] -->

---

## 6. Results

*All values in this section are drawn from models/metadata.json. Fairness and calibration metrics await eval_report.json (run `make evaluate`).*

### 6.1 Discrimination Performance

<!-- PLACEHOLDER: Present a table with rows = {XGBoost, Logistic Regression} and columns = {Split Mode, ROC-AUC, PR-AUC, Brier Score}. Populate from metadata.json:

| Model | Split | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| XGBoost | Time | 0.7464 | 0.4963 | 0.1409 |
| XGBoost | Random | 0.7403 | 0.4689 | 0.1542 |
| XGBoost | Future holdout | 0.7808 | 0.5289 | 0.1373 |
| LR baseline | Time | 0.7506 | 0.5308 | 0.1861 |
| LR baseline | Random | 0.7483 | 0.5141 | 0.1905 |

[metadata.json: xgboost.time_split_metrics, xgboost.random_split_metrics, xgboost.future_holdout_metrics, logistic_regression.time_split_metrics, logistic_regression.random_split_metrics]

Add 1–2 sentences on what the near-parity between time-split and random-split means (expected under IID construction) and why the future-holdout XGBoost edge matters for the backtest framing. -->

### 6.2 Calibration

<!-- PLACEHOLDER: Report ECE and Brier score from eval_report.json. Include a placeholder for the calibration curve figure (reliability diagram). Note that isotonic calibration was applied and describe the observed improvement over uncalibrated scores if available. [TODO: pull ECE from eval_report.json] [TODO: generate calibration curve figure] -->

### 6.3 Fairness Audit

<!-- PLACEHOLDER: Table of demographic parity gap and equalized odds gap by protected group (gender, region). Interpret results in context: synthetic data archetypes deliberately encode differential default rates, so some gap is expected by construction — the audit establishes a measurement baseline, not a fairness guarantee. [TODO: pull all values from eval_report.json] -->

### 6.4 Feature Importance

<!-- PLACEHOLDER: Describe top SHAP features from the trained model. Note whether mobile-money signals or agroclimatic signals dominate, and whether any result is surprising. [TODO: pull SHAP summary from a representative batch run or add a SHAP summary plot export to evaluate.py] -->

---

## 7. Discussion

### 7.1 What Synthetic Data Can Validate

<!-- PLACEHOLDER: Enumerate what this benchmark credibly establishes: the feature schema is complete and coherent, the preprocessing pipeline is correct, the calibration machinery works, the SHAP attribution layer functions, the API returns well-formed responses, and the fairness audit infrastructure is in place. These are necessary preconditions for a real-data experiment. -->

### 7.2 What Synthetic Data Cannot Validate

<!-- PLACEHOLDER: Enumerate the honest limits: IID construction means no real concept drift, no seasonal default cycles, no macroeconomic shocks, no distribution mismatch between training cohorts and deployment population. The archetype distributions are informed by published research but not fitted to real transaction data. ROC-AUC of 0.75 on synthetic data tells us nothing about ROC-AUC on real Tanzanian borrowers. -->

### 7.3 Why the Retrospective Backtest Is the Central Validation Step

<!-- PLACEHOLDER: Argue that the retrospective lender backtest — applying the trained model to a real SACCO/MFI's historical loan portfolio with known outcomes — is the methodologically essential step. The synthetic benchmark is a rigorous scaffold, not a substitute. Frame the near-identical time-split and random-split performance as expected given IID construction, and explain why the future-holdout result (ROC-AUC 0.781 [metadata.json: xgboost.future_holdout_metrics.roc_auc]) is the most credible estimate of deployment performance among the three tiers, subject to the IID caveat above. -->

---

## 8. Limitations

<!-- PLACEHOLDER: Concise enumerated list:
1. All data is synthetic and IID by construction; no real default outcomes are used.
2. Archetype distributions are approximate, informed by published research, not fitted to real Tanzanian transaction data.
3. No bureau or loan-repayment history data — the model is intentionally thin-file-only, but this limits the upper bound on predictive performance.
4. Single-country scope (Tanzania); feature schema may require adaptation for other East African markets.
5. No regulatory approval pathway modeled; Tanzania PDPA (2022) compliance is acknowledged but not implemented.
6. Gender is the only individual protected attribute available in the synthetic schema; ethnicity, disability, and religion are absent.
7. Fairness audit uses archetype as a proxy group, not a real protected class — archetype assignment is known at training time but would not be in deployment.
[CITE: Tanzania Personal Data Protection Act 2022 for item 5] -->

---

## 9. Future Work

### 9.1 Real-Data Retrospective Backtest Protocol

<!-- PLACEHOLDER: Describe the target validation: apply the v0 model to a Tanzanian SACCO or MFI's historical loan portfolio (anonymized, under data-sharing agreement) with known repayment outcomes. Key requirements: borrower mobile money consent, minimum 12-month outcome window, stratified by season and region. Timeline target: Day 90 of this project. [FACT: describe any in-progress lender outreach without naming specific institutions unless confirmed] -->

### 9.2 Regulatory and Data Governance Considerations

<!-- PLACEHOLDER: Note that any transition from synthetic to real data triggers obligations under Tanzania's Personal Data Protection Act (2022) [CITE: PDPA 2022], including purpose limitation, consent, data minimization, and algorithmic transparency requirements. Outline the data governance scaffolding that would be required before a real-data experiment. -->

### 9.3 Feature Schema Extensions

<!-- PLACEHOLDER: Three extensions beyond v0 scope: (1) satellite-derived agroclimatic features (CHIRPS rainfall, Sentinel-2 NDVI) replacing synthetic proxies; (2) seasonal loan product modeling with variable term structures; (3) group-lending dynamics for SACCOs where repayment is partly collective. Each is noted as future work, not v0. -->

### 9.4 Model Extensions

<!-- PLACEHOLDER: Note the deliberate choice to stay with XGBoost for v0 (interpretability, lender trust, tabular-data performance). Future directions could include survival models for time-to-default, conformal prediction for coverage guarantees, and multi-task learning if repayment-amount data becomes available. -->

---

## 10. Conclusion

<!-- PLACEHOLDER: Restate in 2–3 sentences: this paper introduces an open feature schema, a behavioral archetype-based synthetic benchmark, and a three-tier evaluation protocol for alternative-data credit scoring in Tanzania. Synthetic results (XGBoost ROC-AUC 0.746 on time-split holdout [metadata.json: xgboost.time_split_metrics.roc_auc]) validate the methodology scaffold. The central claim — that mobile-money behavior and agroclimatic signals can reliably score smallholder farmers without bureau data — remains to be tested on real data, and the retrospective lender backtest is the next essential step. -->

---

## Acknowledgements

<!-- PLACEHOLDER: Acknowledge NC A&T advisor(s), SROP/Michigan affiliation, and any institutional support. Do not name specific lender contacts until consent is confirmed. -->

---

*Word count target: ~4,000–5,000 words (8–12 pages at standard conference formatting). Current document: scaffold only.*
