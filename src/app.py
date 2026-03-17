from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure repository root is on sys.path so `import src.*` works when the app
# is launched via `streamlit run src/app.py` (Streamlit sets sys.path to the
# script directory by default).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.roi import policy_curve, recommend_target_count

st.set_page_config(page_title="Causal Uplift ROI Dashboard", layout="wide")

ARTIFACT_DIR = Path("artifacts")
SCORED_PATH = ARTIFACT_DIR / "scored_customers.csv"
QINI_PATH = ARTIFACT_DIR / "qini_curve.csv"
DECILE_PATH = ARTIFACT_DIR / "uplift_by_decile.csv"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

st.title("Causal Uplift Modeling + ROI Dashboard")
st.caption("Decision-focused ML for targeting interventions by expected incremental impact.")

if not all(p.exists() for p in [SCORED_PATH, QINI_PATH, DECILE_PATH, METRICS_PATH]):
    st.warning("Artifacts not found. Run the data simulation and training steps first.")
    st.code("python -m src.data_simulation --config configs/base.yaml\npython -m src.train --config configs/base.yaml")
    st.stop()

scored = pd.read_csv(SCORED_PATH)
qini = pd.read_csv(QINI_PATH)
deciles = pd.read_csv(DECILE_PATH)
with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)

business = metrics["business"]

st.sidebar.header("Business assumptions")
conversion_value = st.sidebar.number_input("Incremental conversion value", min_value=1.0, value=float(business["conversion_value"]), step=5.0)
treatment_cost = st.sidebar.number_input("Treatment cost per customer", min_value=0.0, value=float(business["treatment_cost"]), step=1.0)
budget = st.sidebar.number_input("Campaign budget", min_value=0.0, value=float(business["default_budget"]), step=1000.0)

policy = policy_curve(scored, uplift_col="pred_uplift", conversion_value=conversion_value, treatment_cost=treatment_cost)
recommendation = recommend_target_count(policy, budget=budget)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test rows", f"{metrics['rows_test']:,}")
c2.metric("Avg predicted uplift", f"{metrics['avg_pred_uplift_test']:.3f}")
c3.metric("Approx. Qini AUC", f"{metrics['approx_qini_auc']:.2f}")
c4.metric("Recommended targets", f"{recommendation['recommended_customers']:,}")

c5, c6, c7 = st.columns(3)
c5.metric("Expected cost", f"${recommendation['expected_cost']:,.0f}")
c6.metric("Expected incremental value", f"${recommendation['expected_incremental_value']:,.0f}")
c7.metric("Expected net value", f"${recommendation['expected_net_value']:,.0f}")

tab1, tab2, tab3, tab4 = st.tabs(["Policy curve", "Qini curve", "Deciles", "Top targets"])

with tab1:
    fig = px.line(policy, x="customer_index", y="cum_net_value", title="Expected cumulative net value by targeting depth")
    fig.add_vline(x=recommendation["recommended_customers"])
    st.plotly_chart(fig, use_container_width=True)

    fig_cost = px.line(policy, x="customer_index", y=["cum_cost", "cum_incremental_value"], title="Cumulative cost vs expected incremental value")
    st.plotly_chart(fig_cost, use_container_width=True)

with tab2:
    fig_qini = px.line(qini, x="share_targeted", y="incremental_outcomes", title="Approximate Qini curve")
    st.plotly_chart(fig_qini, use_container_width=True)

with tab3:
    st.dataframe(deciles, use_container_width=True)
    fig_decile = px.bar(deciles.sort_values("decile"), x="decile", y=["avg_pred_uplift", "observed_uplift"], barmode="group", title="Predicted vs observed uplift by decile")
    st.plotly_chart(fig_decile, use_container_width=True)

with tab4:
    top_n = st.slider("Top customers to preview", min_value=10, max_value=500, value=50, step=10)
    preview = policy.head(top_n)[["customer_id", "pred_uplift", "p_if_treated", "p_if_control", "expected_net_value"]]
    st.dataframe(preview, use_container_width=True)

st.markdown("### Consulting interpretation")
st.markdown(
    f"""
Under the current assumptions, the model recommends targeting approximately
**{recommendation['recommended_customers']:,} customers** within a **${budget:,.0f}** budget.

At that depth, expected campaign cost is **${recommendation['expected_cost']:,.0f}**, expected incremental value is
**${recommendation['expected_incremental_value']:,.0f}**, and expected net value is **${recommendation['expected_net_value']:,.0f}**.
"""
)

st.markdown(
    "The business point is not just predicting conversion probability. It is ranking customers by "
    "**incremental response to treatment**, which better supports intervention design and budget allocation."
)
