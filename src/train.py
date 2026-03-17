from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.evaluation import approximate_qini_auc, qini_curve_frame, uplift_by_decile
from src.roi import policy_curve, recommend_target_count
from src.uplift import default_dr_learner, default_t_learner, default_x_learner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train uplift model and write artifacts.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--learner", choices=["t", "x", "dr"], help="Override learner type (t, x, or dr).")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Train and compare all learners listed in config.model.compare_learners (defaults to single learner).",
    )
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
    compare_list = cfg["model"].get("compare_learners", [])

    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        random_state=cfg["project"]["random_state"],
        stratify=df[[treatment_col, outcome_col]],
    )

    def build_learner(kind: str) -> object:
        factory: dict[str, Callable[..., object]] = {
            "t": default_t_learner,
            "x": default_x_learner,
            "dr": default_dr_learner,
        }
        if kind not in factory:
            raise ValueError(f"Unknown learner type {kind}")
        return factory[kind](
            feature_columns=feature_columns,
            treatment_column=treatment_col,
            outcome_column=outcome_col,
            random_state=cfg["project"]["random_state"],
        )

    if args.learner:
        learner_types = [args.learner]
    elif args.compare:
        learner_types = compare_list or ["t", "x", "dr"]
    else:
        learner_types = [cfg["model"].get("learner_type", "t")]

    business = cfg["business"]
    metrics_by_model: dict[str, dict] = {}
    artifacts_cache: dict[str, dict[str, object]] = {}

    for learner_type in learner_types:
        learner = build_learner(learner_type)
        learner.fit(train_df)
        scored_test = learner.predict_components(test_df)

        qini_df = qini_curve_frame(scored_test, score_col="pred_uplift", treatment_col=treatment_col, outcome_col=outcome_col)
        decile_df = uplift_by_decile(scored_test, score_col="pred_uplift", treatment_col=treatment_col, outcome_col=outcome_col)
        policy_df = policy_curve(
            scored_test,
            uplift_col="pred_uplift",
            conversion_value=business["conversion_value"],
            treatment_cost=business["treatment_cost"],
        )
        recommendation = recommend_target_count(policy_df, budget=business["default_budget"])

        metrics_by_model[learner_type] = {
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "avg_pred_uplift_test": float(scored_test["pred_uplift"].mean()),
            "approx_qini_auc": float(approximate_qini_auc(qini_df)),
            "recommended_customers_at_default_budget": recommendation["recommended_customers"],
            "expected_net_value_at_default_budget": recommendation["expected_net_value"],
        }

        suffix = f"_{learner_type}"
        learner.save(str(artifact_dir / f"{learner_type}_learner.joblib"))
        scored_sorted = scored_test.sort_values("pred_uplift", ascending=False)
        scored_sorted.to_csv(artifact_dir / f"scored_customers{suffix}.csv", index=False)
        qini_df.to_csv(artifact_dir / f"qini_curve{suffix}.csv", index=False)
        decile_df.to_csv(artifact_dir / f"uplift_by_decile{suffix}.csv", index=False)
        scored_sorted[[id_col, "pred_uplift", "p_if_treated", "p_if_control", treatment_col, outcome_col]].head(100).to_csv(
            artifact_dir / f"sample_scores_preview{suffix}.csv", index=False
        )
        artifacts_cache[learner_type] = {
            "scored": scored_sorted,
            "qini": qini_df,
            "deciles": decile_df,
        }

    primary = max(metrics_by_model.items(), key=lambda kv: kv[1]["approx_qini_auc"])[0]
    # Backward-compatible artifact names for the best-performing learner
    artifacts_cache[primary]["scored"].to_csv(artifact_dir / "scored_customers.csv", index=False)
    artifacts_cache[primary]["qini"].to_csv(artifact_dir / "qini_curve.csv", index=False)
    artifacts_cache[primary]["deciles"].to_csv(artifact_dir / "uplift_by_decile.csv", index=False)
    src_model = artifact_dir / f"{primary}_learner.joblib"
    dst_model = artifact_dir / "t_learner.joblib"
    if src_model.resolve() != dst_model.resolve():
        shutil.copy(src_model, dst_model)
    artifacts_cache[primary]["scored"][[id_col, "pred_uplift", "p_if_treated", "p_if_control", treatment_col, outcome_col]].head(100).to_csv(
        artifact_dir / "sample_scores_preview.csv", index=False
    )

    summary = {
        "feature_columns": feature_columns,
        "business": business,
        "primary_learner": primary,
        "learners": metrics_by_model,
    }
    with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
