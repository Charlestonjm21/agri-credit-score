"""Streamlit demo dashboard for the agri-credit-score v0 model.

Two modes:
- Single borrower: enter features, see score + SHAP explanation
- Batch: upload a parquet/CSV, see score distribution and segment metrics

Run: streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import pickle

import pandas as pd
import plotly.express as px
import streamlit as st

from agri_credit_score.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_LABELS,
    MODELS_DIR,
    risk_band,
)
from agri_credit_score.models.explain import Explainer

st.set_page_config(
    page_title="agri-credit-score — Tanzania demo",
    page_icon="🌾",
    layout="wide",
)


@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "xgb.pkl"
    metadata_path = MODELS_DIR / "metadata.json"
    if not model_path.exists():
        return None, None, None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    return model, Explainer(model), metadata


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    X = df[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


def single_borrower_view(model, explainer) -> None:
    st.header("Score a single borrower")
    st.caption(
        "Enter the borrower's mobile money, agri-context, and loan details to see a "
        "calibrated default probability with explainability."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Mobile money (last 90 days)")
        tx_count = st.number_input("Transaction count", 0, 1000, 140, step=5)
        inflow = st.number_input("Total inflow (TZS)", 0, 50_000_000, 2_400_000, step=50_000)
        outflow = st.number_input("Total outflow (TZS)", 0, 50_000_000, 2_300_000, step=50_000)
        balance = st.number_input("Mean balance (TZS)", 0, 1_000_000, 55_000, step=5_000)
        balance_cv = st.slider("Balance volatility (CV)", 0.0, 3.0, 0.45, 0.05)
        counterparties = st.number_input("Unique counterparties", 0, 200, 20)
        recurrence = st.slider("Recurrence score", 0.0, 1.0, 0.42, 0.01)
        night_ratio = st.slider("Night transaction ratio", 0.0, 1.0, 0.06, 0.01)
        weekend_ratio = st.slider("Weekend transaction ratio", 0.0, 1.0, 0.22, 0.01)
        days_since = st.number_input("Days since last transaction", 0, 365, 2)
        active_days = st.slider("Active days (of 90)", 0, 90, 68)

    with col2:
        st.subheader("Agri context")
        region = st.selectbox(
            "Region",
            ["TZ-01", "TZ-02", "TZ-03", "TZ-04", "TZ-05", "TZ-06", "TZ-07", "TZ-08", "TZ-18"],
            index=0,
        )
        crop = st.selectbox(
            "Primary crop",
            ["maize", "coffee", "beans", "rice", "cotton", "horticulture", "other"],
        )
        rain30 = st.slider("Rainfall anomaly (30d, σ)", -2.0, 2.0, 0.0, 0.1)
        rain90 = st.slider("Rainfall anomaly (90d, σ)", -2.0, 2.0, 0.05, 0.1)
        ndvi = st.slider("Current NDVI", 0.0, 1.0, 0.62, 0.01)
        ndvi_trend = st.slider("NDVI trend (30d)", -0.3, 0.3, 0.02, 0.01)
        price_trend = st.slider("Price trend (30d)", -0.5, 0.5, 0.05, 0.01)
        price_vol = st.slider("Price volatility (30d)", 0.0, 1.0, 0.12, 0.01)
        days_harvest = st.number_input("Days to harvest", 0, 365, 45)
        season = st.selectbox("Season phase", ["planting", "growing", "harvest", "off_season"], 1)

    with col3:
        st.subheader("Borrower & loan")
        age = st.number_input("Age", 18, 100, 34)
        gender = st.selectbox("Gender", ["F", "M", "Other"])
        years = st.number_input("Years farming", 0, 60, 12)
        prior_loans = st.number_input("Prior loans count", 0, 30, 2)
        loan_amount = st.number_input(
            "Loan amount (TZS)", 10_000, 50_000_000, 500_000, step=50_000
        )
        loan_term = st.selectbox("Loan term (days)", [90, 120, 180, 270, 365], index=2)

    if st.button("Score borrower", type="primary"):
        row = {
            "mm_tx_count_90d": tx_count,
            "mm_inflow_total_90d": inflow,
            "mm_outflow_total_90d": outflow,
            "mm_net_flow_90d": inflow - outflow,
            "mm_balance_mean_90d": balance,
            "mm_balance_cv_90d": balance_cv,
            "mm_unique_counterparties_90d": counterparties,
            "mm_recurrence_score": recurrence,
            "mm_night_tx_ratio": night_ratio,
            "mm_weekend_tx_ratio": weekend_ratio,
            "mm_days_since_last_tx": days_since,
            "mm_active_days_90d": active_days,
            "region_code": region,
            "primary_crop": crop,
            "rainfall_anomaly_30d": rain30,
            "rainfall_anomaly_90d": rain90,
            "ndvi_current": ndvi,
            "ndvi_trend_30d": ndvi_trend,
            "commodity_price_trend_30d": price_trend,
            "commodity_price_volatility_30d": price_vol,
            "days_to_harvest": days_harvest,
            "season_phase": season,
            "age": age,
            "gender": gender,
            "years_farming": years,
            "prior_loans_count": prior_loans,
            "loan_amount_tzs": loan_amount,
            "loan_term_days": loan_term,
        }
        X = _prep_df(pd.DataFrame([row]))
        prob = float(model.predict_proba(X)[0, 1])
        band = risk_band(prob)
        attributions = explainer.explain(X, top_k=5)[0]

        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            color = {"Low": "green", "Moderate": "gold", "Elevated": "orange", "High": "red"}[band]
            st.markdown("### Default probability")
            st.markdown(f"<h1 style='color:{color}'>{prob:.1%}</h1>", unsafe_allow_html=True)
            st.markdown(f"**Risk band:** :{color}[{band}]")

        with c2:
            st.markdown("### Top factors in this decision")
            att_df = pd.DataFrame(attributions)
            att_df["abs_shap"] = att_df["shap_value"].abs()
            att_df = att_df.sort_values("abs_shap")
            fig = px.bar(
                att_df,
                x="shap_value",
                y="label",
                color="direction",
                color_discrete_map={
                    "increases_risk": "#d62728",
                    "decreases_risk": "#2ca02c",
                },
                orientation="h",
                labels={"shap_value": "SHAP contribution", "label": ""},
            )
            fig.update_layout(height=350, showlegend=True, legend_title="")
            st.plotly_chart(fig, use_container_width=True)


def batch_view(model, explainer) -> None:
    st.header("Batch scoring")
    st.caption(
        "Upload a parquet or CSV file with borrower features. The model will score every row "
        "and show the distribution, risk-band breakdown, and a sample of the most-risky borrowers."
    )

    uploaded = st.file_uploader("Upload parquet or CSV", type=["parquet", "csv"])
    if uploaded is None:
        return

    try:
        if uploaded.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded)
        else:
            df = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read file: {exc}")
        return

    missing = set(ALL_FEATURES) - set(df.columns)
    if missing:
        st.error(f"Missing required feature columns: {sorted(missing)}")
        return

    X = _prep_df(df)
    df = df.copy()
    df["default_probability"] = model.predict_proba(X)[:, 1]
    df["risk_band"] = df["default_probability"].apply(risk_band)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Probability distribution")
        fig = px.histogram(df, x="default_probability", nbins=40)
        fig.update_layout(height=320, bargap=0.02)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Risk band breakdown")
        band_counts = (
            df["risk_band"]
            .value_counts()
            .reindex(["Low", "Moderate", "Elevated", "High"], fill_value=0)
            .reset_index()
        )
        band_counts.columns = ["risk_band", "count"]
        fig = px.bar(
            band_counts,
            x="risk_band",
            y="count",
            color="risk_band",
            color_discrete_map={
                "Low": "#2ca02c",
                "Moderate": "#ffd700",
                "Elevated": "#ff7f0e",
                "High": "#d62728",
            },
        )
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    if "defaulted" in df.columns:
        st.subheader("Performance on labeled data")
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

        y = df["defaulted"].to_numpy()
        p = df["default_probability"].to_numpy()
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("ROC-AUC", f"{roc_auc_score(y, p):.3f}")
        mc2.metric("PR-AUC", f"{average_precision_score(y, p):.3f}")
        mc3.metric("Brier score", f"{brier_score_loss(y, p):.3f}")

    st.subheader("Top 20 highest-risk borrowers in this batch")
    display_cols = [
        "default_probability",
        "risk_band",
        "region_code",
        "primary_crop",
        "loan_amount_tzs",
        "loan_term_days",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df.sort_values("default_probability", ascending=False).head(20)[display_cols],
        use_container_width=True,
    )

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scored dataset (CSV)",
        csv,
        file_name="scored_borrowers.csv",
        mime="text/csv",
    )


def model_info_view(metadata: dict) -> None:
    st.header("Model information")
    if not metadata:
        st.info("No model metadata found. Train a model with `make train` first.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model version", metadata.get("model_version", "—"))
    c2.metric("Training rows", f"{metadata.get('n_train_rows', 0):,}")
    xgb_metrics = metadata.get("xgboost", {}).get("test_metrics", {})
    c3.metric("Test ROC-AUC", f"{xgb_metrics.get('roc_auc', 0):.3f}")
    c4.metric("Test Brier", f"{xgb_metrics.get('brier', 0):.3f}")

    lr_metrics = metadata.get("logistic_regression", {}).get("test_metrics", {})
    st.subheader("XGBoost vs. logistic regression baseline")
    comp_df = pd.DataFrame(
        {
            "Metric": ["ROC-AUC", "PR-AUC", "Brier"],
            "XGBoost": [
                xgb_metrics.get("roc_auc", 0),
                xgb_metrics.get("pr_auc", 0),
                xgb_metrics.get("brier", 0),
            ],
            "Logistic Regression": [
                lr_metrics.get("roc_auc", 0),
                lr_metrics.get("pr_auc", 0),
                lr_metrics.get("brier", 0),
            ],
        }
    )
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.subheader("Trained at")
    st.code(metadata.get("trained_at", "unknown"))

    st.subheader("Feature list")
    features = metadata.get("features", [])
    st.caption(f"{len(features)} features")
    feat_df = pd.DataFrame(
        [{"feature": f, "label": FEATURE_LABELS.get(f, f)} for f in features]
    )
    st.dataframe(feat_df, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("🌾 agri-credit-score")
    st.caption(
        "v0 demo — synthetic data only. Research prototype for alternative-data credit "
        "scoring of Tanzanian smallholder farmers and agri-SMEs."
    )

    model, explainer, metadata = load_model()
    if model is None:
        st.error(
            "No trained model found. Run the following first:\n\n"
            "```\nmake data\nmake train\n```"
        )
        return

    tab1, tab2, tab3 = st.tabs(["Single borrower", "Batch scoring", "Model info"])
    with tab1:
        single_borrower_view(model, explainer)
    with tab2:
        batch_view(model, explainer)
    with tab3:
        model_info_view(metadata)


if __name__ == "__main__":
    main()
