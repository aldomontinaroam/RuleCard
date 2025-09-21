from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from typing import Union, Optional, Literal
import numpy as np
import pandas as pd

def _to_index(X: Union[pd.DataFrame, np.ndarray]) -> pd.Index:
    return X.columns if hasattr(X, "columns") else pd.Index([f"f{i}" for i in range(X.shape[1])])

def get_feature_importances(
    method: Literal[
        "select_kbest", "select_kbest_chi2", "select_kbest_mi",
          "random_forest", "lightgbm",
        "logistic_regression_l2", "logistic_regression_l1",
        "rfe_logistic_l2", "rfe_logistic_l1", "rfe_lgbm"
    ],
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    top_k: Optional[int] = None,
    importance_type: str = "gain",
    step_ratio: float = 0.10
) -> pd.Series:

    feat_idx = _to_index(X)

    p = X.shape[1]
    step = max(1, int(round(step_ratio * p)))

    # scaling
    methods_need_scaling = {
        "logistic_regression_l2",
        "logistic_regression_l1",
        "rfe_logistic_l2",
        "rfe_logistic_l1",
        "select_kbest_chi2",
        "select_kbest"
    }

    if method in methods_need_scaling:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    # METHODS

    # filter-based
    # - univariate
    if method == "select_kbest":
        selector = SelectKBest(score_func=f_classif, k="all")
        selector.fit(X, y)
        scores = selector.scores_
        if scores is None:
            raise RuntimeError("SelectKBest non ha restituito punteggi.")
        importances = pd.Series(np.nan_to_num(scores, nan=0.0), index=feat_idx)

    elif method == "select_kbest_mi":
        selector = SelectKBest(score_func=mutual_info_classif, k="all")
        selector.fit(X, y)
        importances = pd.Series(selector.scores_, index=feat_idx)

    elif method == "select_kbest_chi2":
        selector = SelectKBest(score_func=chi2, k="all")
        selector.fit(X, y)
        importances = pd.Series(selector.scores_, index=feat_idx)
    
    # embedded approaches
    elif method == "random_forest":
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=feat_idx)

    elif method == "lightgbm":
        lgbm = LGBMClassifier(n_estimators=300, random_state=0, verbose=-1)
        lgbm.fit(X, y)
        booster = lgbm.booster_
        importances = pd.Series(booster.feature_importance(importance_type=importance_type),
                                index=feat_idx)
    
    elif method == "xgboost":
        xgb = XGBClassifier(n_estimators=300, random_state=0, verbosity=0, use_label_encoder=False)
        xgb.fit(X, y)
        importances = pd.Series(xgb.feature_importances_, index=feat_idx)

    elif method == "catboost":
        cat = CatBoostClassifier(iterations=100, random_seed=0, verbose=False)
        cat.fit(X, y)
        importances = pd.Series(cat.feature_importances_, index=feat_idx)

    elif method == "logistic_regression_l2":
        lr = LogisticRegression(penalty="l2", solver="liblinear", max_iter=100)
        lr.fit(X, y)
        importances = pd.Series(np.abs(lr.coef_).ravel(), index=feat_idx)

    elif method == "logistic_regression_l1":
        lr = LogisticRegression(penalty="l1", solver="liblinear", max_iter=100)
        lr.fit(X, y)
        importances = pd.Series(np.abs(lr.coef_).ravel(), index=feat_idx)
    
    # wrapper-based
    elif method in {"rfe_logistic_l2", "rfe_logistic_l1", "rfe_lgbm"}:
        if method == "rfe_logistic_l2":
            base = LogisticRegression(
                penalty="l2",
                solver="saga",
                max_iter=100,
                n_jobs=-1,
                random_state=0
            )
        elif method == "rfe_logistic_l1":
            base = LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=100,
                n_jobs=-1,
                random_state=0
            )
        else:
            base = LGBMClassifier(
                n_estimators=100,
                num_leaves=15,
                max_depth=4,
                n_jobs=-1,
                random_state=0,
                verbose=-1
            )

        selector = RFE(estimator=base, step=step)
        selector.fit(X, y)
        ranks = selector.ranking_
        importances = pd.Series((ranks.max() + 1) - ranks, index=feat_idx)

    else:
        valid_methods = [
            "select_kbest",  "random_forest", "lightgbm",
            "logistic_regression_l2", "logistic_regression_l1",
            "rfe_logistic_l2", "rfe_logistic_l1", "rfe_lgbm", "xgboost", 
            "catboost","select_kbest_chi2", "select_kbest_mi", "select_kbest_chi2"
        ]
        raise ValueError(f"Metodo non riconosciuto: {method!r}. Valori validi: {valid_methods}")

    importances = importances.fillna(0).sort_values(ascending=False)
    if top_k is not None and top_k > 0:
        return importances.head(top_k)
    return importances