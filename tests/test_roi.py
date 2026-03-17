import pandas as pd

from src.roi import add_expected_value_columns, policy_curve, recommend_target_count


def test_expected_value_columns():
    df = pd.DataFrame({"pred_uplift": [0.10, 0.20, -0.05]})
    out = add_expected_value_columns(df, conversion_value=100.0, treatment_cost=10.0)
    assert list(out["expected_incremental_value"].round(2)) == [10.0, 20.0, -5.0]
    assert list(out["expected_net_value"].round(2)) == [0.0, 10.0, -15.0]


def test_recommend_target_count_under_budget():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "pred_uplift": [0.30, 0.20, -0.10],
            "p_if_treated": [0.5, 0.4, 0.2],
            "p_if_control": [0.2, 0.2, 0.3],
        }
    )
    policy = policy_curve(df, conversion_value=100.0, treatment_cost=10.0)
    rec = recommend_target_count(policy, budget=20.0)
    assert rec["recommended_customers"] == 2
    assert rec["expected_cost"] == 20.0
