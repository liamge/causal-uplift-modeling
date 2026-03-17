from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.evaluation import approximate_qini_auc, qini_curve_frame, uplift_by_decile
from src.roi import policy_curve, recommend_target_count
from src.uplift import default_t_learner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train uplift model and write artifacts.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_path = Path(cfg["paths"]["data_path"])
    artifact_dir = Path(cfg["paths"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    feature_columns = cfg["model"]["feature_columns"]
    treatment_col = cfg["model"]["treatment_column"]
    outcome_col = cfg["model"]["outcome_column"]
    id_col = cfg["model"]["id_column"]
    train_size = cfg["model"]["train_size"]

    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        random_state=cfg["project"]["random_state"],
        stratify=df[[treatment_col, outcome_col]],
    )

    learner = default_t_learner(
        feature_columns=feature_columns,
        treatment_column=treatment_col,
        outcome_column=outcome_col,
        random_state=cfg["project"]["random_state"],
    )
    learner.fit(train_df)
    scored_test = learner.predict_components(test_df)

    qini_df = qini_curve_frame(scored_test, score_col="pred_uplift", treatment_col=treatment_col, outcome_col=outcome_col)
    decile_df = uplift_by_decile(scored_test, score_col="pred_uplift", treatment_col=treatment_col, outcome_col=outcome_col)

    business = cfg["business"]
    policy_df = policy_curve(scored_test, uplift_col="pred_uplift", conversion_value=business["conversion_value"], treatment_cost=business["treatment_cost"])
    recommendation = recommend_target_count(policy_df, budget=business["default_budget"])

    metrics = {
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "avg_pred_uplift_test": float(scored_test["pred_uplift"].mean()),
        "approx_qini_auc": float(approximate_qini_auc(qini_df)),
        "recommended_customers_at_default_budget": recommendation["recommended_customers"],
        "expected_net_value_at_default_budget": recommendation["expected_net_value"],
        "feature_columns": feature_columns,
        "business": business,
    }

    learner.save(str(artifact_dir / "t_learner.joblib"))
    scored_test.sort_values("pred_uplift", ascending=False).to_csv(artifact_dir / "scored_customers.csv", index=False)
    qini_df.to_csv(artifact_dir / "qini_curve.csv", index=False)
    decile_df.to_csv(artifact_dir / "uplift_by_decile.csv", index=False)
    with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    scored_test[[id_col, "pred_uplift", "p_if_treated", "p_if_control", treatment_col, outcome_col]].head(100).to_csv(
        artifact_dir / "sample_scores_preview.csv", index=False
    )

    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
