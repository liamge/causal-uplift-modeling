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

# 1) Generate data (pick one)
python -m src.data_simulation --config configs/base.yaml                      # synthetic demo
python -m src.data_marketing --config configs/marketing.yaml                  # real marketing features + semi-synthetic uplift

# 2) Train uplift models and write artifacts (compares T/X/DR by default)
python -m src.train --config configs/base.yaml --compare
python -m src.train --config configs/marketing.yaml --compare

# 3) Launch frontend dashboard (uses artifacts directory; override with ARTIFACT_DIR)
streamlit run src/app.py

# 4) Serve API (separate process)
uvicorn src.api:app --reload --port 8000
```
Outputs land in `data/` and `artifacts/`. The app will warn you if artifacts are missing.

## Modeling & Metrics
- Learners: T-Learner (baseline), X-Learner, and Doubly-Robust Learner. Training compares all three and promotes the best Qini AUC to primary artifacts.
- Metrics written to `artifacts/metrics.json`, including approximate Qini AUC and average predicted uplift.
- Evaluation artifacts: `qini_curve_<learner>.csv`, `uplift_by_decile_<learner>.csv`, and `scored_customers_<learner>.csv`; the best model is also copied to legacy names without suffixes.
- Policy A/B simulator: compares top-uplift targeting vs random within the same budget using expected ROI math.

## Dashboard Walkthrough (`streamlit run src/app.py`)
- Adjust `conversion_value`, `treatment_cost`, and `budget` in the sidebar to see how ROI changes.
- **Policy curve**: cumulative net value by targeting depth with a recommended cutoff marker.
- **Qini curve**: visual check of uplift separation quality.
- **Deciles**: predicted vs observed uplift stability.
- **Top targets**: inspect high-ROI customers and probabilities.
- **Policy A/B**: contrast uplift-driven targeting vs random assignment at the same budget.
- **Model comparison**: table of Qini AUCs across T/X/DR learners.

## API (FastAPI)
- Run `uvicorn src.api:app --reload --port 8000`.
- `GET /health` – simple readiness check.
- `GET /metrics` – returns metrics.json (including primary learner + comparison table).
- `POST /score` – `{"instances": [{feature: value, ...}], "learner": "t|x|dr"}` returns uplift + class probabilities.
- `GET /policies/recommendation?budget=50000&conversion_value=250&treatment_cost=25` – ROI-aware targeting recommendation.
- `POST /policies/ab` – simulate policy A (top uplift) vs policy B (random) for a given budget/cost/value.

## Configuration knobs
- `configs/base.yaml` centralizes feature list, random seeds, train/test split, and business economics.
- `configs/marketing.yaml` builds a semi-synthetic treatment/outcome layer on top of Kaggle's `marketing_campaign.csv` so uplift learning can run on real feature distributions.
- Swap in other data by pointing `paths.data_path` to your CSV and aligning `model.feature_columns`.

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
