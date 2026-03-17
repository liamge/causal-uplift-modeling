from __future__ import annotations

import json
import os
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

from src.roi import policy_curve, recommend_target_count, simulate_policy_ab

st.set_page_config(page_title="Causal Uplift ROI Dashboard", layout="wide")

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

st.title("Causal Uplift Modeling + ROI Dashboard")
st.caption("Decision-focused ML for targeting interventions by expected incremental impact.")

if not METRICS_PATH.exists():
    st.warning("Artifacts not found. Run the data preparation and training steps first.")
    st.code(
        "python -m src.data_simulation --config configs/base.yaml\n"
        "python -m src.train --config configs/base.yaml --compare\n"
        "# or marketing variant\n"
        "python -m src.data_marketing --config configs/marketing.yaml\n"
        "python -m src.train --config configs/marketing.yaml --compare"
    )
    st.stop()

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)

business = metrics["business"]

# Backward compatibility: if metrics.json was produced by older single-learner pipeline,
# synthesize a learners dict so the UI still works.
if "learners" in metrics:
    learners_dict = metrics["learners"]
else:
    legacy_keys = [
        "rows_train",
        "rows_test",
        "avg_pred_uplift_test",
        "approx_qini_auc",
        "recommended_customers_at_default_budget",
        "expected_net_value_at_default_budget",
    ]
    legacy_metrics = {k: metrics[k] for k in legacy_keys if k in metrics}
    learners_dict = {"t": legacy_metrics}
    metrics.setdefault("primary_learner", "t")

learner_options = list(learners_dict.keys())
primary = metrics.get("primary_learner", learner_options[0])
default_index = learner_options.index(primary) if primary in learner_options else 0

learner_choice = st.sidebar.selectbox("Learner", options=learner_options, index=default_index, help="Compare T/X/DR uplift learners")

def _artifact_path(filename: str) -> Path:
    path = ARTIFACT_DIR / f"{filename}_{learner_choice}.csv"
    if path.exists():
        return path
    return ARTIFACT_DIR / f"{filename}.csv"

SCORED_PATH = _artifact_path("scored_customers")
QINI_PATH = _artifact_path("qini_curve")
DECILE_PATH = _artifact_path("uplift_by_decile")

scored = pd.read_csv(SCORED_PATH)
qini = pd.read_csv(QINI_PATH)
deciles = pd.read_csv(DECILE_PATH)

st.sidebar.header("Business assumptions")
conversion_value = st.sidebar.number_input("Incremental conversion value", min_value=1.0, value=float(business["conversion_value"]), step=5.0)
treatment_cost = st.sidebar.number_input("Treatment cost per customer", min_value=0.0, value=float(business["treatment_cost"]), step=1.0)
budget = st.sidebar.number_input("Campaign budget", min_value=0.0, value=float(business["default_budget"]), step=1000.0)
policy_size = int(budget // treatment_cost) if treatment_cost > 0 else len(scored)
random_seed = st.sidebar.number_input("Random seed (A/B sim)", min_value=0, value=42, step=1)

policy = policy_curve(scored, uplift_col="pred_uplift", conversion_value=conversion_value, treatment_cost=treatment_cost)
recommendation = recommend_target_count(policy, budget=budget)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Test rows", f"{learners_dict[learner_choice].get('rows_test', len(scored)):,}")
c2.metric("Avg predicted uplift", f"{learners_dict[learner_choice].get('avg_pred_uplift_test', scored['pred_uplift'].mean()):.3f}")
c3.metric("Approx. Qini AUC", f"{learners_dict[learner_choice].get('approx_qini_auc', float('nan')):.2f}")
c4.metric("Recommended targets", f"{recommendation['recommended_customers']:,}")

c5, c6, c7 = st.columns(3)
c5.metric("Expected cost", f"${recommendation['expected_cost']:,.0f}")
c6.metric("Expected incremental value", f"${recommendation['expected_incremental_value']:,.0f}")
c7.metric("Expected net value", f"${recommendation['expected_net_value']:,.0f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Policy curve", "Qini curve", "Deciles", "Top targets", "Policy A/B"])

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

with tab5:
    st.markdown("Simulate two policies with the same budget to quantify uplift-driven ROI.")
    ab_df = simulate_policy_ab(
        scored,
        uplift_col="pred_uplift",
        conversion_value=conversion_value,
        treatment_cost=treatment_cost,
        budget=budget,
        policy_size=policy_size,
        random_state=int(random_seed),
    )
    st.dataframe(ab_df, use_container_width=True)
    fig_ab = px.bar(ab_df, x="policy", y="expected_net_value", title="Expected net value: uplift vs random")
    st.plotly_chart(fig_ab, use_container_width=True)

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

st.markdown("### Model comparison")
comparison_rows = []
for key, vals in learners_dict.items():
    row = {"learner": key}
    row.update(vals)
    comparison_rows.append(row)
comparison_df = pd.DataFrame(comparison_rows).sort_values("approx_qini_auc", ascending=False)
st.dataframe(comparison_df, use_container_width=True)
