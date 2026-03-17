from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    # Basic cleaning: drop obvious missing incomes/dates
    df = df.dropna(subset=["Income", "Dt_Customer"]).copy()
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%Y-%m-%d", errors="coerce")
    df = df[df["Dt_Customer"].notna()]
    df = df.rename(columns=str.lower)
    df["customer_id"] = df["id"]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prior_accepts = out[["acceptedcmp1", "acceptedcmp2", "acceptedcmp3", "acceptedcmp4", "acceptedcmp5"]].sum(axis=1)
    out["prior_accepts"] = prior_accepts
    out["accepted_any_prior"] = prior_accepts > 0
    out["response_last_campaign"] = out["response"]
    out["campaign_touch_total"] = prior_accepts + 1  # include current campaign
    out["children"] = out["kidhome"] + out["teenhome"]
    out["total_spend"] = (
        out["mntwines"]
        + out["mntfruits"]
        + out["mntmeatproducts"]
        + out["mntfishproducts"]
        + out["mntsweetproducts"]
        + out["mntgoldprods"]
    )
    out["high_recency"] = (out["recency"] <= 30).astype(int)
    out["web_loyal"] = (out["numwebpurchases"] >= 5).astype(int)
    out["store_loyal"] = (out["numstorepurchases"] >= 5).astype(int)
    out["deal_engaged"] = (out["numdealspurchases"] >= 3).astype(int)
    out["web_heavy"] = (out["numwebvisitsmonth"] >= 8).astype(int)
    out["catalog_engaged"] = (out["numcatalogpurchases"] >= 3).astype(int)
    out["revenue_band"] = out["z_revenue"]  # kept for completeness (mostly constant but requested)

    # Education one-hot (top categories only to keep dimensionality small)
    for edu in ["PhD", "Master", "Graduation"]:
        col = f"education_{edu.lower()}"
        out[col] = (out["education"] == edu).astype(int)
    # Marital coarse flags
    out["married"] = out["marital_status"].isin(["Married", "Together"]).astype(int)
    out["single"] = out["marital_status"].isin(["Single", "Divorced", "Widow"]).astype(int)
    return out


def build_semi_synthetic_uplift(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    base_logit = (
        -2.3
        + 0.000025 * df["income"]
        + 0.018 * (df["total_spend"] / 100.0)
        + 0.12 * df["web_loyal"]
        + 0.10 * df["store_loyal"]
        + 0.07 * df["catalog_engaged"]
        - 0.18 * df["web_heavy"]
        + 0.16 * df["deal_engaged"]
        - 0.12 * (df["children"] >= 2).astype(int)
        + 0.08 * df["married"]
        + 0.25 * df["response_last_campaign"]  # ground truth signal from historical response
        + 0.12 * df["prior_accepts"]
    )

    propensity = sigmoid(
        -1.3
        + 0.00002 * df["income"]
        + 0.35 * df["high_recency"]
        + 0.25 * df["deal_engaged"]
        + 0.15 * df["web_loyal"]
        - 0.18 * (df["children"] >= 2).astype(int)
        + 0.22 * df["prior_accepts"]
    )
    treatment = rng.binomial(1, propensity)

    treatment_effect = (
        0.45 * df["high_recency"]
        + 0.3 * (df["income"] <= 40000).astype(int)
        + 0.28 * df["deal_engaged"]
        + 0.22 * df["education_graduation"]
        + 0.16 * df["education_master"]
        - 0.25 * df["web_heavy"]
        - 0.18 * (df["children"] >= 2).astype(int)
        + 0.30 * df["accepted_any_prior"].astype(int)
        + 0.35 * df["response_last_campaign"]
        + 0.18 * df["prior_accepts"]
        + 0.14 * df["catalog_engaged"]
    )

    logit = base_logit + treatment * treatment_effect
    prob = sigmoid(logit)
    outcome = rng.binomial(1, prob)

    out = df.copy()
    out["treatment"] = treatment
    out["outcome"] = outcome
    out["true_uplift_logit"] = treatment_effect
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare semi-synthetic uplift-ready marketing campaign dataset.")
    parser.add_argument("--config", type=str, default="configs/marketing.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_path = Path(cfg["paths"]["raw_marketing_path"])
    output_path = Path(cfg["paths"]["data_path"])
    rng_state = cfg["project"]["random_state"]

    raw = load_raw(data_path)
    engineered = engineer_features(raw)
    uplift_ready = build_semi_synthetic_uplift(engineered, random_state=rng_state)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    uplift_ready.to_csv(output_path, index=False)
    print(f"Wrote marketing uplift dataset to {output_path} with {len(uplift_ready):,} rows.")


if __name__ == "__main__":
    main()
