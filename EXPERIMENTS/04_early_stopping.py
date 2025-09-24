from typing import Callable, Iterable, Optional, Tuple, List, Sequence
import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array
import math
import warnings

class Model(BaseEstimator, ClassifierMixin):
    """
    Additive boosting on 1D trees.
    selection='static': follows a fixed feature order (like Model).
    selection='greedy': chooses at each iteration the feature that minimizes the metric (like Model_v2).
    """
    def __init__(
        self,
        max_depth: int = 2,
        lr: float = 1.0,
        selection: str = "static", # 'static' | 'greedy'
        feature_order: Optional[Iterable] = None,
        feature_importance_fn: Optional[Callable] = None, # es: lambda X,y: (order, importances)
        metric: str = "mse", # 'mse' | 'logloss' for greedy selection
        threshold: float = 0.5,
        random_state: Optional[int] = None,
        epsilon: float = 1e-12,
    ):
        self.max_depth = max_depth
        self.lr = lr
        self.selection = selection
        self.feature_order = list(feature_order) if feature_order is not None else None
        self.feature_importance_fn = feature_importance_fn
        self.metric = metric
        self.threshold = threshold
        self.random_state = random_state
        self.epsilon = epsilon

        self.models_: List[Tuple[int, DecisionTreeRegressor]] = []
        self.p0_: float = None
        self.log_odds_p0_: float = None
        self.used_features_: set = set()
        self.feature_names_in_: Optional[np.ndarray] = None
        self.classes_ = np.array([0, 1])

    def _as_array(self, X):
        import pandas as pd
        if hasattr(X, "values") and not isinstance(X, np.ndarray):
            self.feature_names_in_ = np.asarray(X.columns)
            return X.values, {name: j for j, name in enumerate(self.feature_names_in_)}
        elif isinstance(X, np.ndarray):
            self.feature_names_in_ = np.arange(X.shape[1])
            return X, {j: j for j in range(X.shape[1])}
        else:
            arr = np.asarray(X)
            self.feature_names_in_ = np.arange(arr.shape[1])
            return arr, {j: j for j in range(arr.shape[1])}

    def _init_base_score(self, y: np.ndarray):
        p0 = float(np.clip(np.mean(y), self.epsilon, 1 - self.epsilon))
        self.p0_ = p0
        self.log_odds_p0_ = float(np.log(p0 / (1 - p0)))

    def _greedy_score(self, y, new_F):
        p = expit(new_F)
        if self.metric == "mse":
            return float(np.mean((y - p) ** 2))
        elif self.metric == "logloss":
            p = np.clip(p, self.epsilon, 1 - self.epsilon)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        else:
            raise ValueError(f"metric not supported: {self.metric}")

    def fit(self, X: ArrayLike, y: ArrayLike):
        X_arr, colmap = self._as_array(X)
        y = np.asarray(y).astype(float)
        if not np.array_equal(np.unique(y), [0, 1]) and not np.array_equal(np.unique(y), [0]) and not np.array_equal(np.unique(y), [1]):
            raise ValueError("This classifier supports binary y {0,1}.")

        n_samples, n_features = X_arr.shape
        self.models_ = []
        self.used_features_ = set()

        self._init_base_score(y)
        F = np.full(n_samples, self.log_odds_p0_, dtype=float)

        rng = np.random.RandomState(self.random_state)

        if self.selection not in {"static", "greedy"}:
            raise ValueError("selection must be 'static' or 'greedy'.")

        if self.selection == "static":
            if self.feature_order is None:
                if self.feature_importance_fn is None:
                    order_idx = list(range(n_features))
                else:

                    out = self.feature_importance_fn(X, y)
                    order = out[0] if isinstance(out, (tuple, list)) else out
                    order_idx = [colmap[o] if o in colmap else int(o) for o in order]
            else:
                order_idx = [colmap[o] if o in colmap else int(o) for o in self.feature_order]

            order_idx = order_idx[:n_features]

            for i in order_idx:
                tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
                res = y - expit(F)
                tree.fit(X_arr[:, [i]], res)
                F += self.lr * tree.predict(X_arr[:, [i]])
                self.models_.append((i, tree))
                self.used_features_.add(i)

        else:
            for _ in range(n_features):
                best_score = np.inf
                best_feat = None
                best_tree = None

                res = y - expit(F)

                for i in range(n_features):
                    if i in self.used_features_:
                        continue
                    tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
                    tree.fit(X_arr[:, [i]], res)
                    pred = tree.predict(X_arr[:, [i]])
                    new_F = F + self.lr * pred
                    score = self._greedy_score(y, new_F)
                    if score < best_score:
                        best_score = score
                        best_feat = i
                        best_tree = tree

                if best_feat is None:
                    break

                F += self.lr * best_tree.predict(X_arr[:, [best_feat]])
                self.models_.append((best_feat, best_tree))
                self.used_features_.add(best_feat)

        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        X = np.asarray(X)
        F = np.full(X.shape[0], self.log_odds_p0_, dtype=float)
        for i, tree in self.models_:
            F += self.lr * tree.predict(X[:, [i]])
        return F

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        F = self.decision_function(X)
        p1 = expit(F)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: ArrayLike):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def get_used_features(self) -> List:
        if self.feature_names_in_ is None:
            return [i for i, _ in self.models_]
        return [self.feature_names_in_[i] for i, _ in self.models_]

class ModelES(BaseEstimator, ClassifierMixin):
    """
    Additive boosting on 1D trees.
    selection='static': follows a fixed feature order (like Model).
    selection='greedy': chooses at each iteration the feature that minimizes the metric (like Model_v2).
    
    Early stopping (if early_stopping=True):
      - maximizes the early_stopping_metric
      - stops if: (A) no improvement > tol for patience iterations
                 (B) worsening compared to the previous iteration for patience iterations
    """

    def __init__(
        self,
        max_depth: int = 2,
        lr: float = 1.0,
        selection: str = "greedy", # 'static' | 'greedy'
        feature_order: Optional[Sequence] = None,
        feature_importance_fn: Optional[Callable] = None,
        greedy_metric: str = "logloss", # 'mse' | 'logloss'
        threshold: float = 0.5,
        random_state: Optional[int] = None,
        epsilon: float = 1e-12,
        
        # Early stopping
        early_stopping: bool = False,
        early_stopping_metric: str = "auc", # 'auc' | 'logloss' | 'accuracy' | 'precision' | 'recall' | 'f1'
        validation_fraction: float = 0.1,
        patience: int = 10,
        tol: float = 1e-4,
        validation_X: Optional[ArrayLike] = None,
        validation_y: Optional[ArrayLike] = None,
        verbose: int = 0,
    ):
        self.max_depth = max_depth
        self.lr = lr
        self.selection = selection
        self.feature_order = list(feature_order) if feature_order is not None else None
        self.feature_importance_fn = feature_importance_fn
        self.greedy_metric = greedy_metric
        self.threshold = threshold
        self.random_state = random_state
        self.epsilon = epsilon

        self.models_: List[Tuple[int, DecisionTreeRegressor]] = []
        self.p0_: Optional[float] = None
        self.log_odds_p0_: Optional[float] = None
        self.used_features_: List[int] = []
        self.n_estimators_: int = 0
        self.classes_: np.ndarray = np.array([0, 1])
        self.feature_names_in_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None


        self.early_stopping = early_stopping
        self.early_stopping_metric = early_stopping_metric
        self.validation_fraction = validation_fraction
        self.patience = int(patience)
        self.tol = tol
        self.validation_X = validation_X
        self.validation_y = validation_y
        self.verbose = verbose
        self.validation_scores_: List[float] = []
        self.best_iteration_: Optional[int] = None
        self.best_score_: Optional[float] = None
        self.stop_reason_: Optional[str] = None   # "no_improve", "worsening", None

        if self.selection == 'static':
            if self.feature_order is None and self.feature_importance_fn is None:
                raise ValueError("Insert feature_order or feature_importance_fn")

    def _rng(self):
        return np.random.RandomState(self.random_state)

    def _init_base_score(self, y: np.ndarray):
        p0 = float(np.clip(np.mean(y), self.epsilon, 1 - self.epsilon))
        self.p0_ = p0
        self.log_odds_p0_ = float(np.log(p0 / (1 - p0)))

    @staticmethod
    def _score_from_margin(F: np.ndarray, y: np.ndarray, metric: str, thr: float, eps: float) -> float:
        """Evaluates a classification metric given the margin F (logit)."""
        p = expit(F)
        if metric == "auc":
            if len(np.unique(y)) < 2:
                return 0.5
            return roc_auc_score(y, p)
        elif metric == "logloss":
            p = np.clip(p, eps, 1 - eps)
            return -log_loss(y, p)
        else:
            preds = (p >= thr).astype(int)
            if metric == "accuracy":  return accuracy_score(y, preds)
            if metric == "precision": return precision_score(y, preds, zero_division=0)
            if metric == "recall":    return recall_score(y, preds, zero_division=0)
            if metric == "f1":        return f1_score(y, preds, zero_division=0)
            raise ValueError(f"early_stopping_metric not supported: {metric}")

    @staticmethod
    def _greedy_loss(y: np.ndarray, F_new: np.ndarray, kind: str, eps: float) -> float:
        """Loss for greedy feature selection."""
        p = expit(F_new)
        if kind == "mse":
            return float(np.mean((y - p) ** 2))
        elif kind == "logloss":
            p = np.clip(p, eps, 1 - eps)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))  # logloss
        else:
            raise ValueError(f"greedy_metric not supported: {kind}")

    def _as_array(self, X):
        import pandas as pd
        if hasattr(X, "values") and not isinstance(X, np.ndarray):
            self.feature_names_in_ = np.asarray(X.columns)
            return X.values, {name: j for j, name in enumerate(self.feature_names_in_)}
        elif isinstance(X, np.ndarray):
            self.feature_names_in_ = np.arange(X.shape[1])
            return X, {j: j for j in range(X.shape[1])}
        else:
            arr = np.asarray(X)
            self.feature_names_in_ = np.arange(arr.shape[1])
            return arr, {j: j for j in range(arr.shape[1])}

    # ---------- API ----------
    def fit(self, X: ArrayLike, y: ArrayLike):
        if self.feature_importance_fn is not None and self.selection != "static":
            warnings.warn("feature_importance_fn in input: setting selection='static'.")
            self.selection = "static"

        X_in = X
        X_arr, colmap = self._as_array(X_in)

        X, y = check_X_y(X_arr, y, accept_sparse=False, dtype=float)
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError("Questo classificatore supporta y binaria {0,1}.")

        rng = self._rng()
        n, p = X.shape
        self.models_.clear()
        self.used_features_.clear()
        self.validation_scores_.clear()
        self.n_estimators_ = 0
        self.stop_reason_ = None

        if self.early_stopping:
            if self.validation_X is not None and self.validation_y is not None:
                X_tr, y_tr = X, y
                X_val = check_array(self.validation_X, dtype=float)
                y_val = np.asarray(self.validation_y, dtype=float)
            else:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X, y, test_size=self.validation_fraction,
                    random_state=self.random_state, stratify=y
                )
        else:
            X_tr, y_tr = X, y
            X_val = y_val = None

        self._init_base_score(y_tr)
        F_tr = np.full(y_tr.shape[0], self.log_odds_p0_, dtype=float)
        F_val = np.full(y_val.shape[0], self.log_odds_p0_, dtype=float) if self.early_stopping else None

        best_score = None
        best_iter = 0
        prev_score = None
        if self.early_stopping:
            s0 = self._score_from_margin(F_val, y_val, self.early_stopping_metric, self.threshold, self.epsilon)
            self.validation_scores_.append(s0)
            best_score = s0
            prev_score = s0
            if self.verbose:
                print(f"Iter 0: val_{self.early_stopping_metric}={s0:.6f}")

        T = p

        if self.selection == "static":
            order_idx = None

            if self.feature_importance_fn is not None:
                out = self.feature_importance_fn(X_in, y)

                def _map_to_idx(seq_like):
                    idxs = []
                    seen = set()
                    for o in list(seq_like):
                        if o in colmap:
                            j = int(colmap[o])
                        else:
                            try:
                                j = int(o)
                            except Exception:
                                continue
                        if 0 <= j < p and j not in seen:
                            seen.add(j)
                            idxs.append(j)
                    return idxs

                order_candidate = None
                importances = None

                if isinstance(out, (tuple, list)) and len(out) >= 1:
                    order_candidate = out[0]
                    if len(out) >= 2:
                        importances = out[1]
                else:
                    order_candidate = out

                is_series_like = hasattr(order_candidate, "values") and hasattr(order_candidate, "index")
                if isinstance(order_candidate, dict) or is_series_like:
                    if isinstance(order_candidate, dict):
                        items = sorted(order_candidate.items(), key=lambda kv: kv[1], reverse=True)
                    else:
                        try:
                            items = list(order_candidate.sort_values(ascending=False).items())
                        except Exception:
                            items = [(k, v) for k, v in zip(list(order_candidate.index), list(order_candidate.values))]
                            items.sort(key=lambda kv: kv[1], reverse=True)
                    order_idx = _map_to_idx([k for k, _ in items])

                if order_idx is None:
                    try:
                        seq = list(order_candidate)
                        are_numbers = all(isinstance(v, (int, float, np.number)) for v in seq)
                        if len(seq) == p and are_numbers:
                            order_idx = list(np.argsort(np.asarray(seq))[::-1])
                    except TypeError:
                        pass

                if order_idx is None:
                    try:
                        order_idx = _map_to_idx(order_candidate)
                    except Exception:
                        order_idx = None

                if order_idx is None or len(order_idx) == 0:
                    raise ValueError(
                        "feature_importance_fn non ha restituito un ordine valido né un vettore di importanze interpretabile."
                    )

            # 2) self.feature_order oppure ordine naturale
            else:
                if self.feature_order is None:
                    order_idx = list(range(p))
                else:
                    order_idx = []
                    for o in self.feature_order:
                        if o in colmap:
                            order_idx.append(int(colmap[o]))
                        else:
                            order_idx.append(int(o))

            order_idx = order_idx[:T]

        elif self.selection != "greedy":
            raise ValueError("selection deve essere 'static' o 'greedy'.")

        if self.selection == "greedy":
            k = max(1, int(round(math.sqrt(p)))) if not hasattr(self, "feature_subsample") or self.feature_subsample is None \
                else int(min(max(1, self.feature_subsample), p))

        no_improve_rounds = 0
        worsening_rounds = 0

        for t in range(1, T + 1):
            residual = y_tr - expit(F_tr)

            if self.selection == "static":
                i = order_idx[t - 1]
                tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
                tree.fit(X_tr[:, [i]], residual)
                delta_tr = tree.predict(X_tr[:, [i]])
                F_tr = F_tr + self.lr * delta_tr
                chosen = (i, tree)

            else:
                remaining = [j for j in range(p) if j not in self.used_features_]
                if len(remaining) == 0:
                    break
                if len(remaining) > k:
                    cand = self._rng().choice(remaining, size=k, replace=False)
                else:
                    cand = np.array(remaining, dtype=int)

                best_loss = np.inf
                chosen = None
                best_pred = None
                for i in cand:
                    tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
                    tree.fit(X_tr[:, [i]], residual)
                    pred = tree.predict(X_tr[:, [i]])
                    loss = self._greedy_loss(y_tr, F_tr + self.lr * pred, self.greedy_metric, self.epsilon)
                    if loss < best_loss:
                        best_loss = loss
                        chosen = (i, tree)
                        best_pred = pred

                F_tr = F_tr + self.lr * best_pred

            i, tree = chosen
            self.models_.append((i, tree))
            self.used_features_.append(i)

            if self.early_stopping:
                F_val = F_val + self.lr * tree.predict(X_val[:, [i]])
                val_score = self._score_from_margin(F_val, y_val, self.early_stopping_metric, self.threshold, self.epsilon)
                self.validation_scores_.append(val_score)

                if self.verbose:
                    print(f"Iter {t}: val_{self.early_stopping_metric}={val_score:.6f}")

                improved = (best_score is None) or (val_score > best_score + self.tol)
                worsened = (prev_score is not None) and (val_score < prev_score - self.tol)

                if improved:
                    best_score = val_score
                    best_iter = t
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1

                if worsened:
                    worsening_rounds += 1
                else:
                    worsening_rounds = 0

                if no_improve_rounds >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at iter {t}: no improvement for {self.patience} iterations."
                              f"(best_iter={best_iter}, best_score={best_score:.6f})")
                    self.stop_reason_ = "no_improve"
                    break

                if worsening_rounds >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at iter {t}: worsening for {self.patience} iterations."
                              f"(best_iter={best_iter}, best_score={best_score:.6f})")
                    self.stop_reason_ = "worsening"
                    break

                prev_score = val_score

        if self.early_stopping:
            self.models_ = self.models_[:best_iter]
            self.used_features_ = self.used_features_[:best_iter]
            self.best_iteration_ = best_iter
            self.best_score_ = best_score

        self.n_estimators_ = len(self.models_)
        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        X = check_array(X, dtype=float)
        F = np.full(X.shape[0], self.log_odds_p0_, dtype=float)
        for i, tree in self.models_:
            F += self.lr * tree.predict(X[:, [i]])
        return F

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        F = self.decision_function(X)
        p1 = expit(F)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: ArrayLike) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def get_used_features(self) -> List:
        if self.feature_names_in_ is None:
            return [i for i, _ in self.models_]
        return [self.feature_names_in_[i] for i, _ in self.models_]

    def get_early_stopping_info(self) -> dict:
        return {
            "early_stopping_used": self.early_stopping,
            "best_iteration": self.best_iteration_ if self.early_stopping else None,
            "best_score": self.best_score_ if self.early_stopping else None,
            "final_estimators": self.n_estimators_,
            "validation_scores": self.validation_scores_ if self.early_stopping else [],
            "early_stopping_metric": self.early_stopping_metric if self.early_stopping else None,
            "patience": self.patience if self.early_stopping else None,
            "stop_reason": self.stop_reason_ if self.early_stopping else None,
        }


# =========EXPERIMENTS=========

from __future__ import annotations
import math, time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef
)
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon, t as student_t
from scipy.special import expit  # se serve altrove nel progetto

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

def _acc(y, yhat):   return accuracy_score(y, yhat)
def _prec(y, yhat):  return precision_score(y, yhat, zero_division=0)
def _rec(y, yhat):   return recall_score(y, yhat, zero_division=0)
def _f1(y, yhat):    return f1_score(y, yhat, zero_division=0)
def _bacc(y, yhat):  return balanced_accuracy_score(y, yhat)
def _mcc(y, yhat):   return matthews_corrcoef(y, yhat)

_LABEL_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": _acc,
    "precision": _prec,
    "recall": _rec,
    "f1": _f1,
    "balanced_accuracy": _bacc,
    "mcc": _mcc,
}
_METRICS_ALL = ["auc", "logloss"] + list(_LABEL_METRICS.keys())
_METRICS_PRINT = ["auc", "accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc", "logloss"]
_PRETTY = {
    "auc": "AUC", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1": "F1", "balanced_accuracy": "Balanced Acc.", "mcc": "MCC", "logloss": "Log Loss"
}

def _threshold_grid(probs: np.ndarray, max_grid: int = 512) -> np.ndarray:
    p = np.asarray(probs, float)
    thr = np.unique(p if p.size <= max_grid else np.quantile(p, np.linspace(0, 1, max_grid)))
    return np.unique(np.clip(np.r_[0.0, thr, 1.0], 0, 1))

def safe_auc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true)
    return 0.5 if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_prob)

def bootstrap_ci(x, n_boot: int = 10_000, alpha: float = 0.05,
                 random_state: int = 42, stat_fn: Callable = np.mean) -> Tuple[float,float]:
    x = np.asarray(x)
    if x.size == 0: return (np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    if stat_fn in (np.mean, np.median):
        boot = stat_fn(x[idx], axis=1)
    else:
        boot = np.array([stat_fn(x[i]) for i in idx])
    lo, hi = np.quantile(boot, [alpha/2, 1 - alpha/2])
    return float(lo), float(hi)

def _best_thresholds_per_metric(y_val: np.ndarray, p_val: np.ndarray) -> Dict[str, float]:
    yv = np.asarray(y_val, int)
    thr = _threshold_grid(p_val)
    best = {m: (0.5, -np.inf) for m in _LABEL_METRICS}  # (t, score)
    for t in thr:
        yhat = (p_val >= t).astype(int)
        for m, fn in _LABEL_METRICS.items():
            s = fn(yv, yhat)
            if s > best[m][1]:
                best[m] = (float(t), float(s))
    return {m: t for m, (t, _) in best.items()}

def _eval_all(y_test: np.ndarray, p_test: np.ndarray, thr: Dict[str, float]) -> Dict[str, float]:
    out = {"auc": safe_auc(y_test, p_test),
           "logloss": log_loss(y_test, np.clip(p_test, 1e-12, 1 - 1e-12))}
    yt = np.asarray(y_test, int)
    for m, t in thr.items():
        out[m] = _LABEL_METRICS[m](yt, (p_test >= t).astype(int))
    return out

def wilcoxon_safe(x, alternative: str = "two-sided") -> Dict[str, float]:
    x = np.asarray(x, float)
    if x.size == 0 or np.allclose(x, 0): return {"stat": 0.0, "pvalue": 1.0}
    try:
        s, p = wilcoxon(x, alternative=alternative, zero_method="wilcox")
        return {"stat": float(s), "pvalue": float(p)}
    except ValueError:
        return {"stat": np.nan, "pvalue": 1.0}

def tost_paired(diff, delta=0.01, alpha=0.05):
    d = np.asarray(diff, float); n = d.size
    md = float(d.mean()); sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd == 0 or n < 2:
        inside = abs(md) < delta
        return {"equivalent": bool(inside), "pvalue": 0.0 if inside else 1.0,
                "mean_diff": md, "ci95_mean": (md, md), "delta": float(delta)}
    se = sd / math.sqrt(n); df = n - 1
    p1 = 1 - student_t.cdf((md + delta) / se, df=df)
    p2 = student_t.cdf((md - delta) / se, df=df)
    tcrit = student_t.ppf(1 - 0.05, df=df)
    return {"equivalent": bool((p1 < alpha) and (p2 < alpha)),
            "pvalue": float(max(p1, p2)),
            "mean_diff": md,
            "ci95_mean": (md - tcrit * se, md + tcrit * se),
            "delta": float(delta)}

def make_equiv_margins(delta_auc: float, base_other: Optional[Dict] = None) -> Dict:
    other = dict(accuracy=0.01, precision=0.02, recall=0.02, f1=0.02, balanced_accuracy=0.01, mcc=0.02)
    if base_other: other.update(base_other)
    other["auc"] = float(delta_auc)
    return other

@dataclass
class RunConfig:
    max_depth: int = 2
    lr: float = 1.0
    selection: str = "greedy" # "greedy" | "static"
    greedy_metric: str = "logloss"
    random_state: int = 42

@dataclass
class SplitConfig:
    test_size: float = 0.2
    val_size_within_tv: float = 0.25
    stratify: bool = True
    random_state: int = 42

@dataclass
class Experiment:
    name: str
    selection: str # "greedy" | "static"
    extra_model_kwargs: Optional[Dict[str, Any]] = None

def _build_model_kwargs(base: RunConfig, exp: Experiment, n_features: int) -> Dict[str, Any]:
    kwargs = dict(max_depth=base.max_depth, lr=base.lr, selection=exp.selection,
                  random_state=base.random_state, verbose=0)
    if exp.selection == "greedy":
        kwargs["greedy_metric"] = base.greedy_metric
    if exp.extra_model_kwargs: kwargs.update(exp.extra_model_kwargs)
    return kwargs

def _fit_once(X_tr, y_tr, X_val, y_val, X_te, y_te, kwargs: Dict[str, Any], use_es: bool, es_params: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(kwargs)
    params["early_stopping"] = bool(use_es)
    if use_es: params.update(es_params)

    model = ModelES(**params)
    model.validation_X = X_val; model.validation_y = y_val

    tic = time.perf_counter()
    model.fit(X_tr, y_tr)
    t = time.perf_counter() - tic

    p_val  = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_te)[:, 1]
    thr    = _best_thresholds_per_metric(y_val, p_val)
    metrics = _eval_all(y_te, p_test, thr)

    try:
        stages = int(getattr(model, "n_estimators_", 0))
    except Exception:
        stages = 0
    try:
        features = int(len(set(model.get_used_features())))
    except Exception:
        features = np.nan

    return {"time": t, "stages": stages, "features": features, "metrics": metrics}

def _run_dataset(name: str, load_fn: Callable, base: RunConfig, split: SplitConfig,
                 exp: Experiment, es_params: Optional[Dict] = None) -> Dict[str, Any]:
    data = load_fn(name); X, y = data["X"], data["y"]
    strat = y if split.stratify else None
    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=split.test_size,
                                              random_state=split.random_state, stratify=strat)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tv, y_tv, test_size=split.val_size_within_tv,
                                                random_state=split.random_state, stratify=y_tv if strat is not None else None)

    kwargs = _build_model_kwargs(base, exp, n_features=X.shape[1])
    es_def = dict(early_stopping=True, early_stopping_metric="logloss",
                  validation_fraction=0.2, patience=10, tol=1e-4)
    if es_params: es_def.update(es_params)

    out_es   = _fit_once(X_tr, y_tr, X_val, y_val, X_te, y_te, kwargs, True,  es_def)
    out_noes = _fit_once(X_tr, y_tr, X_val, y_val, X_te, y_te, kwargs, False, es_def)

    row = {"dataset": name, "selection": exp.name,
           "time_es": out_es["time"], "time_noes": out_noes["time"],
           "stages_es": out_es["stages"], "stages_noes": out_noes["stages"],
           "features_es": out_es["features"], "features_noes": out_noes["features"]}
    for m in _METRICS_ALL:
        row[f"{m}_es"] = out_es["metrics"][m]
        row[f"{m}_noes"] = out_noes["metrics"][m]
    return row

def run_benchmark(
    datasets: List[str],
    load_fn: Callable,
    base_cfg: RunConfig,
    split_cfg: SplitConfig,
    experiment: Experiment,
    equiv_margins: Optional[Dict] = None,
    alpha: float = 0.05,
    random_state: int = 42,
    es_params: Optional[Dict] = None,
    with_ci: bool = True,
):
    rows = []
    for name in datasets:
        try:
            rows.append(_run_dataset(name, load_fn, base_cfg, split_cfg, experiment, es_params=es_params))
        except Exception as e:
            print(f"[WARN] Dataset '{name}' ({experiment.name}) saltato: {e}")

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError(f"No results available for {experiment.name}.")

    for m in _METRICS_ALL:
        results[f"d_{m}"] = results[f"{m}_es"] - results[f"{m}_noes"]
    results["d_time"]     = results["time_noes"]   - results["time_es"]
    results["d_stages"]   = results["stages_noes"] - results["stages_es"]
    results["d_features"] = results["features_noes"] - results["features_es"]

    stats = {"per_metric": {}, "equiv_margins_used": (equiv_margins or {})}
    for m in _METRICS_ALL:
        entry = {"wilcoxon_two_sided": wilcoxon_safe(results[f"d_{m}"], "two-sided")}
        if equiv_margins and (m in equiv_margins):
            entry["tost"] = tost_paired(results[f"d_{m}"].values, delta=float(equiv_margins[m]), alpha=alpha)
        stats["per_metric"][m] = entry

    stats["time_one_sided"]     = wilcoxon_safe(results["d_time"], "greater")
    stats["stages_one_sided"]   = wilcoxon_safe(results["d_stages"], "greater")
    stats["features_one_sided"] = wilcoxon_safe(results["d_features"], "greater")

    agg = {}
    for m in _METRICS_ALL:
        agg[f"{m}_ES_mean"]   = results[f"{m}_es"].mean()
        agg[f"{m}_NoES_mean"] = results[f"{m}_noes"].mean()
        agg[f"Δ{m}_mean"]     = results[f"d_{m}"].mean()
    agg.update({
        "Tempo_ES_median(s)": results["time_es"].median(),
        "Tempo_NoES_median(s)": results["time_noes"].median(),
        "ΔTempo_median(s) (NoES-ES)": results["d_time"].median(),
        "Stadi_ES_median": results["stages_es"].median(),
        "Stadi_NoES_median": results["stages_noes"].median(),
        "ΔStadi_median (NoES-ES)": results["d_stages"].median(),
        "Feature_ES_median": results["features_es"].median(),
        "Feature_NoES_median": results["features_noes"].median(),
        "ΔFeature_median (NoES-ES)": results["d_features"].median(),
    })
    stats["aggregates"] = agg

    if with_ci:
        stats["ci_mean_diff_95"] = {m: bootstrap_ci(results[f"d_{m}"].values, stat_fn=np.mean, random_state=random_state)
                                    for m in _METRICS_ALL}
        stats["ci_time_median_diff_95"]     = bootstrap_ci(results["d_time"].values,     stat_fn=np.median, random_state=random_state)
        stats["ci_stages_median_diff_95"]   = bootstrap_ci(results["d_stages"].values,   stat_fn=np.median, random_state=random_state)
        stats["ci_features_median_diff_95"] = bootstrap_ci(results["d_features"].values, stat_fn=np.median, random_state=random_state)

    return results, stats

def _one_setting(exp: Experiment, datasets: List[str], load_fn: Callable, base_cfg: RunConfig, split_cfg: SplitConfig,
                 delta_auc: float, patience: int, tol: float, alpha: float):
    equiv_marg = make_equiv_margins(delta_auc)
    es_params = {"patience": int(patience), "tol": float(tol)}
    res, st = run_benchmark(
        datasets=datasets, load_fn=load_fn,
        base_cfg=base_cfg, split_cfg=split_cfg,
        experiment=exp, equiv_margins=equiv_marg, alpha=alpha, es_params=es_params
    )
    per = st["per_metric"]
    eq_flags = [d.get("tost", {}).get("equivalent", False) for d in per.values() if "tost" in d]
    row = {
        "selection": exp.name, "delta_auc": delta_auc, "patience": patience, "tol": tol,
        "eq_count": int(sum(eq_flags)), "eq_total": int(len(eq_flags)),
        "auc_mean_diff": per.get("auc", {}).get("tost", {}).get("mean_diff", np.nan),
        "time_p_one_sided": st["time_one_sided"]["pvalue"],
        "ΔTempo_median(s)": st["aggregates"]["ΔTempo_median(s) (NoES-ES)"],
        "ΔStadi_median": st["aggregates"]["ΔStadi_median (NoES-ES)"],
        "ΔFeature_median": st["aggregates"]["ΔFeature_median (NoES-ES)"],
    }
    return row, ((exp.name, float(delta_auc), int(patience), float(tol)), (res, st))

def sweep_experiments(
    experiments: List[Experiment],
    datasets: List[str],
    load_fn: Callable,
    base_cfg: RunConfig,
    split_cfg: SplitConfig,
    delta_auc_grid: List[float] = (0.005, 0.01, 0.02),
    patience_grid: List[int] = (3, 5, 10, 20),
    tol_grid: List[float] = (0.0, 1e-4),
    alpha: float = 0.05,
    n_jobs: int = 1,
    prefer_backend: str = "threads",
):
    tasks = [(exp, da, pa, tl) for exp in experiments for da in delta_auc_grid for pa in patience_grid for tl in tol_grid]

    need_threads = any(any(callable(v) for v in (e.extra_model_kwargs or {}).values()) for e in experiments)
    prefer = "threads" if (prefer_backend == "threads" or need_threads) else "processes"

    if _HAS_JOBLIB and (n_jobs is not None) and (n_jobs != 1):
        outs = Parallel(n_jobs=n_jobs, verbose=0, prefer=prefer)(
            delayed(_one_setting)(exp, datasets, load_fn, base_cfg, split_cfg, da, pa, tl, alpha)
            for (exp, da, pa, tl) in tasks
        )
    else:
        outs = [_one_setting(exp, datasets, load_fn, base_cfg, split_cfg, da, pa, tl, alpha) for (exp, da, pa, tl) in tasks]

    rows, by_setting = [], {}
    for row, kv in outs:
        rows.append(row); key, val = kv; by_setting[key] = val

    summary_df = pd.DataFrame(rows).sort_values(
        by=["selection", "eq_count", "ΔStadi_median", "time_p_one_sided"],
        ascending=[True, False, False, True]
    ).reset_index(drop=True)
    return summary_df, by_setting

def _fmt_tuple(t):
    try:    return f"({round(float(t[0]),4)}, {round(float(t[1]),4)})"
    except: return str(t)

def print_detailed_report(results: pd.DataFrame, stats: dict):
    pd.set_option("display.width", 180)
    print("\n=== Aggregated statistics (ES vs NoES) ===")
    rows = [{
        "Metrica": _PRETTY[m],
        "ES_mean": stats["aggregates"].get(f"{m}_ES_mean", np.nan),
        "NoES_mean": stats["aggregates"].get(f"{m}_NoES_mean", np.nan),
        "Δ_mean (ES-NoES)": stats["aggregates"].get(f"Δ{m}_mean", np.nan),
    } for m in _METRICS_PRINT]
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Statistics tests per metric ===")
    eq_hits = 0; eq_total = 0
    for m in _METRICS_PRINT:
        per = stats["per_metric"].get(m, {})
        w = per.get("wilcoxon_two_sided")
        tost = per.get("tost")
        line = f"{_PRETTY[m]:<16} | Wilcoxon: " + (f"stat={w['stat']}, p={w['pvalue']:.4g}" if w else "n/a")
        if tost:
            eq_total += 1
            line += (f" | TOST(Δ={tost['delta']}): equivalent={tost['equivalent']}, "
                     f"p={tost['pvalue']:.4g}, mean_diff={tost['mean_diff']:.4g}, "
                     f"CI95_mean={_fmt_tuple(tost['ci95_mean'])}")
            if tost["equivalent"]: eq_hits += 1
        print(line)

    if eq_total > 0:
        winners = [_PRETTY[m] for m in _METRICS_PRINT
                   if stats['per_metric'].get(m, {}).get('tost', {}).get('equivalent')]
        print(f"\n[Executive summary] TOST on {eq_hits}/{eq_total} metrics ({', '.join(winners)}).")

    print("\n=== Time and Complexity ===")
    print(f"Tempo   (NoES-ES > 0)  → stat={stats['time_one_sided']['stat']}, p={stats['time_one_sided']['pvalue']:.4g} | "
          f"median Δ(s)={stats['aggregates']['ΔTempo_median(s) (NoES-ES)']:.4f}")
    print(f"Stadi   (NoES-ES > 0)  → stat={stats['stages_one_sided']['stat']}, p={stats['stages_one_sided']['pvalue']:.4g} | "
          f"median Δ={stats['aggregates']['ΔStadi_median (NoES-ES)']:.4f}")
    print(f"Feature (NoES-ES > 0)  → stat={stats['features_one_sided']['stat']}, p={stats['features_one_sided']['pvalue']:.4g} | "
          f"median Δ={stats['aggregates']['ΔFeature_median (NoES-ES)']:.4f}")

    cols = ["dataset", "selection"] + [f"d_{m}" for m in _METRICS_PRINT] + ["d_time", "d_stages", "d_features"]
    show_cols = [c for c in cols if c in results.columns]
    print("\n=== Δ per dataset (ES - NoES) ===")
    print(results[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


DATASETS = list(get_available_datasets().keys())

base_cfg = RunConfig(max_depth=2, lr=1.0, selection="greedy", greedy_metric="logloss",
                     random_state=42)
split_cfg = SplitConfig(test_size=0.20, val_size_within_tv=0.20, stratify=True, random_state=42)

experiments = [
    Experiment(name="greedy", selection="greedy"),
    Experiment(name="static", selection="static",
               extra_model_kwargs={"feature_importance_fn": get_feature_importance}),
]

DELTA_AUC = [0.005, 0.01]
PATIENCE  = [5, 10]
TOL       = [1e-4]

summary, by_setting = sweep_experiments(
    experiments=experiments,
    datasets=DATASETS,
    load_fn=load_preprocess,
    base_cfg=base_cfg,
    split_cfg=split_cfg,
    delta_auc_grid=DELTA_AUC,
    patience_grid=PATIENCE,
    tol_grid=TOL,
    alpha=0.05,
    n_jobs=-1,
    prefer_backend="threads"
)

print("\n=== Leaderboard complessivo ===")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

for sel in summary["selection"].unique():
    top = summary[summary["selection"] == sel].iloc[0]
    key = (sel, float(top["delta_auc"]), int(top["patience"]), float(top["tol"]))
    res_best, stats_best = by_setting[key]
    print(f"\n=== MIGLIOR SETTING per selection='{sel}' ===")
    print(f"delta_auc={top['delta_auc']}, patience={int(top['patience'])}, tol={top['tol']}")
    print_detailed_report(res_best, stats_best)
