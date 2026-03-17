from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_simulated_campaign_data(
    n_samples: int = 12000,
    treatment_rate: float = 0.5,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 76, size=n_samples)
    tenure_months = rng.integers(1, 121, size=n_samples)
    monthly_spend = rng.gamma(shape=5.0, scale=20.0, size=n_samples) + 20
    support_tickets_90d = rng.poisson(lam=1.8, size=n_samples)
    email_engagement = np.clip(rng.beta(2, 3, size=n_samples), 0, 1)
    prior_discount_user = rng.binomial(1, 0.35, size=n_samples)

    region = rng.choice(["east", "north", "south", "west"], size=n_samples, p=[0.28, 0.24, 0.26, 0.22])
    segment = rng.choice(["consumer", "smallbiz", "enterprise"], size=n_samples, p=[0.62, 0.25, 0.13])

    region_north = (region == "north").astype(int)
    region_south = (region == "south").astype(int)
    region_west = (region == "west").astype(int)

    segment_smallbiz = (segment == "smallbiz").astype(int)
    segment_enterprise = (segment == "enterprise").astype(int)

    treatment = rng.binomial(1, treatment_rate, size=n_samples)

    base_logit = (
        -1.9
        + 0.015 * (age - 40)
        + 0.012 * np.minimum(tenure_months, 60)
        + 0.0035 * monthly_spend
        - 0.22 * support_tickets_90d
        + 1.4 * email_engagement
        + 0.25 * prior_discount_user
        + 0.10 * region_north
        - 0.08 * region_south
        + 0.06 * region_west
        + 0.22 * segment_smallbiz
        + 0.16 * segment_enterprise
    )

    treatment_effect = (
        0.65 * (email_engagement > 0.45).astype(float)
        + 0.35 * (tenure_months < 24).astype(float)
        + 0.30 * prior_discount_user
        + 0.22 * segment_smallbiz
        + 0.18 * (support_tickets_90d >= 2).astype(float)
        - 0.30 * (tenure_months > 72).astype(float)
        - 0.18 * (monthly_spend > 180).astype(float)
    )

    logit = base_logit + treatment * treatment_effect
    prob = sigmoid(logit)
    outcome = rng.binomial(1, prob, size=n_samples)

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n_samples + 1),
            "age": age,
            "tenure_months": tenure_months,
            "monthly_spend": monthly_spend.round(2),
            "support_tickets_90d": support_tickets_90d,
            "email_engagement": email_engagement.round(4),
            "prior_discount_user": prior_discount_user,
            "region": region,
            "segment": segment,
            "region_north": region_north,
            "region_south": region_south,
            "region_west": region_west,
            "segment_smallbiz": segment_smallbiz,
            "segment_enterprise": segment_enterprise,
            "treatment": treatment,
            "outcome": outcome,
            "true_uplift_logit": treatment_effect.round(4),
        }
    )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic campaign data for uplift modeling.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    sim_cfg = cfg["simulation"]
    random_state = cfg["project"]["random_state"]
    output_path = Path(cfg["paths"]["data_path"])
    df = build_simulated_campaign_data(
        n_samples=sim_cfg["n_samples"],
        treatment_rate=sim_cfg["treatment_rate"],
        random_state=random_state,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df):,} rows to {output_path}")


if __name__ == "__main__":
    main()
