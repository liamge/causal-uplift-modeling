from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_config
from src.roi import policy_curve, recommend_target_count, simulate_policy_ab
from src.uplift import DRLearner, TLearner, XLearner


class ScoreRequest(BaseModel):
    instances: list[dict] = Field(..., description="List of feature dictionaries")
    learner: Optional[Literal["t", "x", "dr"]] = Field(default=None, description="Which learner to use (default primary)")


class ABRequest(BaseModel):
    budget: Optional[float] = Field(None, description="Campaign budget")
    policy_size: Optional[int] = Field(None, description="Number of customers to treat")
    conversion_value: float
    treatment_cost: float
    uplift_col: str = "pred_uplift"
    learner: Optional[Literal["t", "x", "dr"]] = None
    random_seed: int = 42


def load_metrics(artifact_dir: Path) -> dict:
    metrics_path = artifact_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=4)
def load_model(artifact_dir: Path, learner: str) -> TLearner | XLearner | DRLearner:
    path = artifact_dir / f"{learner}_learner.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found for learner '{learner}' at {path}")
    loader_map = {"t": TLearner.load, "x": XLearner.load, "dr": DRLearner.load}
    return loader_map[learner](str(path))


def get_artifact_dir(config_path: Path) -> Path:
    cfg = load_config(config_path)
    return Path(cfg["paths"]["artifact_dir"])


CONFIG_PATH = Path(os.getenv("CONFIG_PATH", "configs/base.yaml")).resolve()
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", get_artifact_dir(CONFIG_PATH))).resolve()

app = FastAPI(title="Causal Uplift API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "artifact_dir": str(ARTIFACT_DIR)}


@app.get("/metrics")
def read_metrics() -> dict:
    return load_metrics(ARTIFACT_DIR)


@app.post("/score")
def score(request: ScoreRequest) -> dict:
    metrics = load_metrics(ARTIFACT_DIR)
    learner = request.learner or metrics.get("primary_learner", "t")
    model = load_model(ARTIFACT_DIR, learner)
    feature_columns = metrics.get("feature_columns")
    rows = request.instances
    df = pd.DataFrame(rows)
    if feature_columns:
        missing = set(feature_columns) - set(df.columns)
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing feature columns: {sorted(missing)}")
        df = df[feature_columns]
    scored = model.predict_components(df)
    return {
        "learner": learner,
        "scores": scored[["pred_uplift", "p_if_treated", "p_if_control"]].to_dict(orient="records"),
    }


@app.get("/policies/recommendation")
def recommend(budget: float, conversion_value: Optional[float] = None, treatment_cost: Optional[float] = None) -> dict:
    metrics = load_metrics(ARTIFACT_DIR)
    business = metrics["business"]
    conv_val = conversion_value if conversion_value is not None else business["conversion_value"]
    treat_cost = treatment_cost if treatment_cost is not None else business["treatment_cost"]

    scored_path = ARTIFACT_DIR / "scored_customers.csv"
    if not scored_path.exists():
        raise HTTPException(status_code=404, detail="scored_customers.csv not found. Train a model first.")
    scored = pd.read_csv(scored_path)
    policy = policy_curve(scored, uplift_col="pred_uplift", conversion_value=conv_val, treatment_cost=treat_cost)
    rec = recommend_target_count(policy, budget=budget)
    return {"budget": budget, "conversion_value": conv_val, "treatment_cost": treat_cost, "recommendation": rec}


@app.post("/policies/ab")
def policy_ab(request: ABRequest) -> dict:
    metrics = load_metrics(ARTIFACT_DIR)
    learner = request.learner or metrics.get("primary_learner", "t")
    suffix = f"_{learner}"
    scored_path = ARTIFACT_DIR / f"scored_customers{suffix}.csv"
    if not scored_path.exists():
        scored_path = ARTIFACT_DIR / "scored_customers.csv"
    if not scored_path.exists():
        raise HTTPException(status_code=404, detail="scored_customers.csv not found. Train a model first.")
    scored = pd.read_csv(scored_path)
    ab = simulate_policy_ab(
        scored,
        uplift_col=request.uplift_col,
        conversion_value=request.conversion_value,
        treatment_cost=request.treatment_cost,
        budget=request.budget,
        policy_size=request.policy_size,
        random_state=request.random_seed,
    )
    return {"learner": learner, "ab_results": ab.to_dict(orient="records")}
