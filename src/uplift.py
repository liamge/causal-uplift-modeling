from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier


@dataclass
class TLearner:
    treatment_model: object
    control_model: object
    feature_columns: list[str]
    treatment_column: str
    outcome_column: str

    def fit(self, df: pd.DataFrame) -> "TLearner":
        treated = df[df[self.treatment_column] == 1].copy()
        control = df[df[self.treatment_column] == 0].copy()
        self.treatment_model.fit(treated[self.feature_columns], treated[self.outcome_column])
        self.control_model.fit(control[self.feature_columns], control[self.outcome_column])
        return self

    def predict_components(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.feature_columns]
        p_treat = self.treatment_model.predict_proba(x)[:, 1]
        p_control = self.control_model.predict_proba(x)[:, 1]
        out = df.copy()
        out["p_if_treated"] = p_treat
        out["p_if_control"] = p_control
        out["pred_uplift"] = p_treat - p_control
        return out

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "TLearner":
        return joblib.load(path)


def default_t_learner(
    feature_columns: list[str],
    treatment_column: str,
    outcome_column: str,
    random_state: int = 42,
) -> TLearner:
    base = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=40,
        random_state=random_state,
        n_jobs=-1,
    )
    return TLearner(
        treatment_model=clone(base),
        control_model=clone(base),
        feature_columns=feature_columns,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
    )
