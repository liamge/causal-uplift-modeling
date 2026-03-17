# Causal Uplift Modeling + ROI Dashboard

Decision-focused ML demo that goes beyond propensity: simulate campaigns, estimate treatment uplift, and turn predictions into ROI-driven targeting recommendations with a Streamlit dashboard.

## Highlights
- Synthetic campaign generator with configurable treatment rate and signal strength (`src/data_simulation.py`).
- Uplift modeling via a **T-Learner** of twin RandomForest classifiers (`src/uplift.py`).
- Business framing baked in: policy curve, budget-aware target recommendation, and expected net value (`src/roi.py`).
- Evaluation with Qini curve, decile tables, and uplift-at-k style metrics (`src/evaluation.py`).
- Streamlit dashboard to tweak cost/value assumptions live and preview top targets (`src/app.py`).

## Project Layout
- `configs/base.yaml` – single source of truth for features, paths, and business assumptions.
- `data/` – simulated campaign CSV (created by the data step).
- `artifacts/` – model, scored customers, metrics, and evaluation curves (created by the train step).
- `src/` – modular code: simulation, training, uplift learner, ROI logic, dashboard.
- `tests/` – lightweight unit tests for ROI math.

## Quickstart (local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Generate data
python -m src.data_simulation --config configs/base.yaml

# 2) Train uplift models and write artifacts
python -m src.train --config configs/base.yaml

# 3) Launch dashboard
streamlit run src/app.py
```
Outputs land in `data/` and `artifacts/`. The app will warn you if artifacts are missing.

## Modeling & Metrics
- Approach: T-Learner (two forests) with uplift = `P(y=1|treated) - P(y=1|control)`.
- Metrics written to `artifacts/metrics.json`, including approximate Qini AUC and average predicted uplift.
- Evaluation artifacts: `qini_curve.csv`, `uplift_by_decile.csv`, and a `scored_customers.csv` preview for the top candidates.

## Dashboard Walkthrough (`streamlit run src/app.py`)
- Adjust `conversion_value`, `treatment_cost`, and `budget` in the sidebar to see how ROI changes.
- **Policy curve**: cumulative net value by targeting depth with a recommended cutoff marker.
- **Qini curve**: visual check of uplift separation quality.
- **Deciles**: predicted vs observed uplift stability.
- **Top targets**: inspect high-ROI customers and probabilities.

## Configuration knobs
- `configs/base.yaml` centralizes feature list, random seeds, train/test split, and business economics.
- Swap in real data by pointing `paths.data_path` to your CSV and aligning `model.feature_columns`.

## Tests
```bash
pytest
```
(`tests/test_roi.py` validates the ROI math.)

## Portfolio blurb
Built an end-to-end uplift modeling system that estimates which customers respond to treatment, then converts those predictions into ROI-based targeting via an interactive dashboard. Demonstrates causal thinking, treatment-effect estimation, and decision-focused ML beyond standard propensity modeling.

## How to publish this as a GitHub repo
1) Create an empty repo on GitHub (no README/license added in the UI). Note the `https://github.com/<user>/<repo>.git` URL.
2) From this project root, initialize and connect:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: causal uplift + ROI dashboard"
   git branch -M main
   git remote add origin https://github.com/<user>/<repo>.git
   git push -u origin main
   ```
3) Add a brief repo description and topic tags like `causal-inference`, `uplift-modeling`, `streamlit`, `portfolio` on GitHub.
