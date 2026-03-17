from __future__ import annotations

import numpy as np
import pandas as pd


def add_uplift_deciles(df: pd.DataFrame, score_col: str = "pred_uplift") -> pd.DataFrame:
    out = df.copy()
    out["uplift_percentile"] = out[score_col].rank(method="first", pct=True)
    out["uplift_decile"] = pd.qcut(out["uplift_percentile"], 10, labels=list(range(1, 11)))
    out["uplift_decile"] = out["uplift_decile"].astype(int)
    return out


def qini_curve_frame(
    df: pd.DataFrame,
    score_col: str = "pred_uplift",
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
) -> pd.DataFrame:
    ordered = df.sort_values(score_col, ascending=False).reset_index(drop=True).copy()
    ordered["cum_treated"] = (ordered[treatment_col] == 1).cumsum()
    ordered["cum_control"] = (ordered[treatment_col] == 0).cumsum()
    ordered["cum_y_treated"] = ((ordered[treatment_col] == 1) * ordered[outcome_col]).cumsum()
    ordered["cum_y_control"] = ((ordered[treatment_col] == 0) * ordered[outcome_col]).cumsum()
    control_rate = ordered["cum_y_control"] / ordered["cum_control"].replace(0, np.nan)
    incremental = ordered["cum_y_treated"] - control_rate.fillna(0.0) * ordered["cum_treated"]
    curve = pd.DataFrame(
        {
            "n_targeted": np.arange(1, len(ordered) + 1),
            "share_targeted": np.arange(1, len(ordered) + 1) / len(ordered),
            "incremental_outcomes": incremental.fillna(0.0),
        }
    )
    return curve


def approximate_qini_auc(curve: pd.DataFrame) -> float:
    return float(np.trapz(curve["incremental_outcomes"], curve["share_targeted"]))


def uplift_by_decile(
    df: pd.DataFrame,
    score_col: str = "pred_uplift",
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
) -> pd.DataFrame:
    scored = add_uplift_deciles(df, score_col=score_col)
    rows = []
    for decile in sorted(scored["uplift_decile"].unique(), reverse=True):
        chunk = scored[scored["uplift_decile"] == decile]
        treated = chunk[chunk[treatment_col] == 1]
        control = chunk[chunk[treatment_col] == 0]
        y_t = treated[outcome_col].mean() if len(treated) else np.nan
        y_c = control[outcome_col].mean() if len(control) else np.nan
        rows.append(
            {
                "decile": int(decile),
                "customers": int(len(chunk)),
                "avg_pred_uplift": float(chunk[score_col].mean()),
                "treated_rate": float(y_t) if pd.notna(y_t) else np.nan,
                "control_rate": float(y_c) if pd.notna(y_c) else np.nan,
                "observed_uplift": float(y_t - y_c) if pd.notna(y_t) and pd.notna(y_c) else np.nan,
            }
        )
    return pd.DataFrame(rows)
