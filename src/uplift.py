from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression


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


@dataclass
class XLearner:
    treatment_outcome_model: object
    control_outcome_model: object
    tau_t_model: object
    tau_c_model: object
    propensity_model: object
    feature_columns: list[str]
    treatment_column: str
    outcome_column: str

    def fit(self, df: pd.DataFrame) -> "XLearner":
        treated = df[df[self.treatment_column] == 1].copy()
        control = df[df[self.treatment_column] == 0].copy()
        x_t, y_t = treated[self.feature_columns], treated[self.outcome_column]
        x_c, y_c = control[self.feature_columns], control[self.outcome_column]

        self.treatment_outcome_model.fit(x_t, y_t)
        self.control_outcome_model.fit(x_c, y_c)

        # Imputed treatment effects
        mu0_on_t = self.control_outcome_model.predict_proba(x_t)[:, 1]
        d1 = y_t - mu0_on_t
        mu1_on_c = self.treatment_outcome_model.predict_proba(x_c)[:, 1]
        d0 = mu1_on_c - y_c

        self.tau_t_model.fit(x_t, d1)
        self.tau_c_model.fit(x_c, d0)
        self.propensity_model.fit(df[self.feature_columns], df[self.treatment_column])
        return self

    def predict_components(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.feature_columns]
        p_treat = self.propensity_model.predict_proba(x)[:, 1]
        mu1 = self.treatment_outcome_model.predict_proba(x)[:, 1]
        mu0 = self.control_outcome_model.predict_proba(x)[:, 1]
        tau_t = self.tau_t_model.predict(x)
        tau_c = self.tau_c_model.predict(x)
        tau = p_treat * tau_c + (1 - p_treat) * tau_t

        out = df.copy()
        out["p_if_treated"] = mu1
        out["p_if_control"] = mu0
        out["pred_uplift"] = tau
        out["propensity"] = p_treat
        return out

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "XLearner":
        return joblib.load(path)


@dataclass
class DRLearner:
    treatment_outcome_model: object
    control_outcome_model: object
    cate_model: object
    propensity_model: object
    feature_columns: list[str]
    treatment_column: str
    outcome_column: str

    def fit(self, df: pd.DataFrame) -> "DRLearner":
        x = df[self.feature_columns]
        t = df[self.treatment_column]
        y = df[self.outcome_column]

        treated = df[t == 1]
        control = df[t == 0]
        self.treatment_outcome_model.fit(treated[self.feature_columns], treated[self.outcome_column])
        self.control_outcome_model.fit(control[self.feature_columns], control[self.outcome_column])
        self.propensity_model.fit(x, t)

        p = self.propensity_model.predict_proba(x)[:, 1].clip(1e-3, 1 - 1e-3)
        mu1 = self.treatment_outcome_model.predict_proba(x)[:, 1]
        mu0 = self.control_outcome_model.predict_proba(x)[:, 1]

        pseudo_outcome = ((t * (y - mu1)) / p) - (((1 - t) * (y - mu0)) / (1 - p)) + mu1 - mu0
        self.cate_model.fit(x, pseudo_outcome)
        return self

    def predict_components(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.feature_columns]
        mu1 = self.treatment_outcome_model.predict_proba(x)[:, 1]
        mu0 = self.control_outcome_model.predict_proba(x)[:, 1]
        tau = self.cate_model.predict(x)
        out = df.copy()
        out["p_if_treated"] = mu1
        out["p_if_control"] = mu0
        out["pred_uplift"] = tau
        out["propensity"] = self.propensity_model.predict_proba(x)[:, 1]
        return out

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "DRLearner":
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


def default_x_learner(
    feature_columns: list[str],
    treatment_column: str,
    outcome_column: str,
    random_state: int = 42,
) -> XLearner:
    outcome_base = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=30,
        random_state=random_state,
        n_jobs=-1,
    )
    tau_base = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        random_state=random_state,
        n_jobs=-1,
    )
    propensity = LogisticRegression(max_iter=500, n_jobs=1)
    return XLearner(
        treatment_outcome_model=clone(outcome_base),
        control_outcome_model=clone(outcome_base),
        tau_t_model=clone(tau_base),
        tau_c_model=clone(tau_base),
        propensity_model=clone(propensity),
        feature_columns=feature_columns,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
    )


def default_dr_learner(
    feature_columns: list[str],
    treatment_column: str,
    outcome_column: str,
    random_state: int = 42,
) -> DRLearner:
    outcome_base = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=30,
        random_state=random_state,
        n_jobs=-1,
    )
    cate = RandomForestRegressor(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=20,
        random_state=random_state,
        n_jobs=-1,
    )
    propensity = LogisticRegression(max_iter=500, n_jobs=1)
    return DRLearner(
        treatment_outcome_model=clone(outcome_base),
        control_outcome_model=clone(outcome_base),
        cate_model=clone(cate),
        propensity_model=clone(propensity),
        feature_columns=feature_columns,
        treatment_column=treatment_column,
        outcome_column=outcome_column,
    )
