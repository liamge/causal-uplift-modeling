from __future__ import annotations

import numpy as np
import pandas as pd


def add_expected_value_columns(
    df: pd.DataFrame,
    uplift_col: str = "pred_uplift",
    conversion_value: float = 180.0,
    treatment_cost: float = 18.0,
) -> pd.DataFrame:
    out = df.copy()
    out["expected_incremental_value"] = out[uplift_col] * conversion_value
    out["expected_net_value"] = out["expected_incremental_value"] - treatment_cost
    return out


def policy_curve(
    df: pd.DataFrame,
    uplift_col: str = "pred_uplift",
    conversion_value: float = 180.0,
    treatment_cost: float = 18.0,
) -> pd.DataFrame:
    ranked = add_expected_value_columns(
        df,
        uplift_col=uplift_col,
        conversion_value=conversion_value,
        treatment_cost=treatment_cost,
    ).sort_values(uplift_col, ascending=False).reset_index(drop=True)
    ranked["customer_index"] = np.arange(1, len(ranked) + 1)
    ranked["cum_cost"] = ranked["customer_index"] * treatment_cost
    ranked["cum_incremental_value"] = ranked["expected_incremental_value"].cumsum()
    ranked["cum_net_value"] = ranked["expected_net_value"].cumsum()
    ranked["share_targeted"] = ranked["customer_index"] / len(ranked)
    return ranked


def recommend_target_count(policy_df: pd.DataFrame, budget: float) -> dict:
    affordable = policy_df[policy_df["cum_cost"] <= budget]
    if affordable.empty:
        return {
            "recommended_customers": 0,
            "expected_cost": 0.0,
            "expected_incremental_value": 0.0,
            "expected_net_value": 0.0,
        }
    best = affordable.loc[affordable["cum_net_value"].idxmax()]
    return {
        "recommended_customers": int(best["customer_index"]),
        "expected_cost": float(best["cum_cost"]),
        "expected_incremental_value": float(best["cum_incremental_value"]),
        "expected_net_value": float(best["cum_net_value"]),
    }
