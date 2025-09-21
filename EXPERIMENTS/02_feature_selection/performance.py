from feature_importance import get_feature_importances
from typing import Union, OrderedDict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)
from sklearn.base import BaseEstimator

def performance_on_topk(
    method: str,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: pd.Series,
    val_size: float = 0.2,
    k: int = 5,
    seed: int = 0,
    include_all: bool = True
) -> OrderedDict[str, float]:

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        stratify=y_train if set(np.unique(y_train)) <= {0, 1} else None,
        random_state=seed,
    )

    ranking = get_feature_importances(method, X_tr, y_tr, top_k=k)
    top_k_features = ranking.index.tolist()

    results: OrderedDict[str, float] = OrderedDict()

    def _eval(clf: BaseEstimator, Xtr, ytr, Xv, yv, prefix: str):
        clf.fit(Xtr, ytr)
        y_pred = clf.predict(Xv)
        proba  = clf.predict_proba(Xv)[:, 1] if hasattr(clf, "predict_proba") else None
        results[f"{prefix}_acc"] = accuracy_score(yv, y_pred)
        results[f"{prefix}_bal_acc"] = balanced_accuracy_score(yv, y_pred)
        results[f"{prefix}_recall"] = recall_score(yv, y_pred)
        results[f"{prefix}_f1"] = f1_score(yv, y_pred)
        results[f"{prefix}_mcc"] = matthews_corrcoef(yv, y_pred)
        results[f"{prefix}_auc"] = roc_auc_score(yv, proba) if proba is not None else float("nan")
        results[f"{prefix}_avgp"] = average_precision_score(yv, proba) if proba is not None else float("nan")

    X_tr_top = X_tr[top_k_features]
    X_val_top = X_val[top_k_features]

    _eval(RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed),
          X_tr_top, y_tr, X_val_top, y_val, "rf_top")

    _eval(LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=seed, verbose=-1),
          X_tr_top, y_tr, X_val_top, y_val, "lgbm_top")

    if include_all:
        _eval(RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed),
              X_tr, y_tr, X_val, y_val, "rf_all")
        _eval(LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=seed, verbose=-1),
              X_tr, y_tr, X_val, y_val, "lgbm_all")

    return results