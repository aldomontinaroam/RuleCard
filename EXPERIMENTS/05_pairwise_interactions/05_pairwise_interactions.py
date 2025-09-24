from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union, List, Sequence

from FAST import FAST

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    roc_auc_score, log_loss,
    accuracy_score, precision_score,
    recall_score, f1_score, precision_recall_curve,
    roc_curve, fbeta_score, balanced_accuracy_score,
    matthews_corrcoef
)
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array
from scipy.special import expit

from joblib import Parallel, delayed
from tqdm import tqdm as _tqdm
import logging

import numpy as np

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

"""
################## PAIRWISE INTERACTIONS ##################
"""
def _sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out

def _logit(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)

@dataclass
class _Stage:
    feats: Tuple[int, ...] # (i,) for 1D, (i,j) for 2D
    tree: DecisionTreeRegressor # pre-trained regressor on residuals
    kind: str # 'uni' or 'pair'
    score: float = 0.0 # gain on train (loss_before - loss_after)

logger = logging.getLogger("Model2D")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                                  datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class Model2D(ClassifierMixin, BaseEstimator):
    """
    New functions:
    - Selection always by gain (no fixed priority to 1D).
    - Pairwise features are always refitted on the current residual before comparison.
    - Candidate cache updated every cache_refresh_every iterations.
    - FAST re-executed every fast_refresh_every iterations (merging new pairs).
    """

    def __init__(
        self,
        max_depth: int = 2,
        lr: float = 1.0,
        selection: str = "greedy",                 # 'greedy' | 'static'
        feature_order: Optional[Sequence] = None,
        feature_importance_fn: Optional[Callable] = None,

        group_mode: str = "mixed",                 # 'univariate' | 'pairwise' | 'mixed'
        n_stages: Optional[int] = None,
        greedy_metric: str = "logloss",            # 'logloss' o 'mse'
        threshold: float = 0.5,
        random_state: Optional[int] = None,
        epsilon: float = 1e-12,

        # Early stopping
        early_stopping: bool = False,
        early_stopping_metric: str = "auc",        # 'auc','logloss','accuracy','precision','recall','f1'
        validation_fraction: float = 0.1,
        patience: int = 10,
        tol: float = 1e-4,
        validation_X: Optional[ArrayLike] = None,
        validation_y: Optional[ArrayLike] = None,

        # Policy
        allow_pair_reuse: bool = False,
        block_univariate_if_in_pair: bool = True,

        # FAST
        fast_bins: int = 16,
        fast_num_threads: int = 1,
        fast_min_support: int = 50,
        fast_max_cells: int = 1024,

        # Refresh
        cache_refresh_every: int = 5,
        fast_refresh_every: int = 10,

        # Dictionary mode
        precompute_univariate: bool = True,
        precompute_pairs: bool = True,
        cache_max_mb: Optional[int] = 1024,
        score_batch: int = 512,
        verbose: int = 0,
    ):
        self._estimator_type = "classifier" 
        self.max_depth = max_depth
        self.lr = lr
        self.selection = selection
        self.feature_order = list(feature_order) if feature_order is not None else None
        self.feature_importance_fn = feature_importance_fn

        self.group_mode = group_mode
        self.n_stages = n_stages
        self.greedy_metric = greedy_metric
        self.threshold = threshold
        self.random_state = random_state
        self.epsilon = epsilon

        self.early_stopping = early_stopping
        self.early_stopping_metric = early_stopping_metric
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.tol = tol
        self.validation_X = validation_X
        self.validation_y = validation_y

        self.allow_pair_reuse = allow_pair_reuse
        self.block_univariate_if_in_pair = block_univariate_if_in_pair

        self.fast_bins = fast_bins
        self.fast_num_threads = fast_num_threads
        self.fast_min_support = fast_min_support
        self.fast_max_cells = fast_max_cells

        self.cache_refresh_every = cache_refresh_every
        self.fast_refresh_every = fast_refresh_every

        self.precompute_univariate = precompute_univariate
        self.precompute_pairs = precompute_pairs
        self.cache_max_mb = cache_max_mb
        self.score_batch = score_batch

        self.verbose = verbose

        self.stages_: List[_Stage] = []
        self.used_features_1d_: List[int] = []
        self.used_pairs_: List[int] = []
        self._used_pair_tuples: set[Tuple[int, int]] = set()
        self._features_used_in_pairs: set[int] = set()
        self.n_estimators_: int = 0
        self.classes_: np.ndarray = np.array([0, 1])
        self.feature_names_in_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

        self.p0_: Optional[float] = None
        self.log_odds_p0_: Optional[float] = None

        self._uni_index_map: Dict[int, int] = {}

        # early stopping info
        self.validation_scores_: List[float] = []
        self.best_iteration_: Optional[int] = None
        self.best_score_: Optional[float] = None

        # dict candidates
        self._Xtr_cache: Optional[np.ndarray] = None
        self._uni_trees: List[DecisionTreeRegressor] = []
        self._pair_meta: List[Tuple[int, int]] = []
        self._pair_cache_meta_idx = None 
        self._pair_trees_cache = None

        # cache predictions su train
        self._uni_pred_cache: Optional[np.ndarray] = None
        self._pair_pred_cache: Optional[np.ndarray] = None

        # pairs from FAST
        self._fast_pairs_: List[Tuple[int, int]] = []
        self.max_pair_candidates: Optional[int] = 256


        if self.selection == 'static':
            if self.feature_order is None and self.feature_importance_fn is None:
                raise ValueError("Insert feature_order or feature_importance_fn")

    def _init_base_score(self, y: np.ndarray):
        p0 = float(np.clip(np.mean(y), self.epsilon, 1 - self.epsilon))
        self.p0_ = p0
        self.log_odds_p0_ = float(np.log(p0 / (1 - p0)))
    
    def _more_tags(self):
        return {"estimator_type": "classifier"}

    @staticmethod
    def _score_from_margin(F: np.ndarray, y: np.ndarray, metric: str, thr: float, eps: float) -> float:
        p = expit(F)
        if metric == "auc":
            if len(np.unique(y)) < 2:
                return 0.5
            return roc_auc_score(y, p)
        elif metric == "logloss":
            p = np.clip(p, eps, 1 - eps)
            return -log_loss(y, p)  # segno negativo massimizziamo
        else:
            preds = (p >= thr).astype(int)
            if metric == "accuracy": return accuracy_score(y, preds)
            if metric == "precision": return precision_score(y, preds, zero_division=0)
            if metric == "recall": return recall_score(y, preds, zero_division=0)
            if metric == "f1": return f1_score(y, preds, zero_division=0)
            raise ValueError(f"early_stopping_metric non supportata: {metric}")

    @staticmethod
    def _loss_scalar(F: np.ndarray, y: np.ndarray, metric: str, eps: float, sample_weight: np.ndarray = None) -> float:
        if metric == "logloss":
            p = expit(F)
            p = np.clip(p, eps, 1 - eps)
            ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            if sample_weight is None: return float(np.mean(ll))
            w = sample_weight / (np.sum(sample_weight) + 1e-12)
            return float(np.sum(w * ll))
        elif metric == "mse":
            p = expit(F)
            return float(np.mean((y - p) ** 2))
        else:
            raise ValueError("greedy_metric deve essere 'logloss' o 'mse'.")

    @staticmethod
    def _logloss_block(F_base, y, pred_block, lr, eps, sample_weight=None):
        M = F_base[:, None] + lr * pred_block
        P = 1.0 / (1.0 + np.exp(-M))
        P = np.clip(P, eps, 1 - eps)
        ll = -(y[:, None] * np.log(P) + (1 - y)[:, None] * np.log(1 - P))
        if sample_weight is None:
            return np.mean(ll, axis=0)
        w = sample_weight / (np.sum(sample_weight) + 1e-12)
        return np.sum(ll * w[:, None], axis=0)

    @staticmethod
    def _mse_block(F_base, y, pred_block, lr, sample_weight=None):
        P = 1.0 / (1.0 + np.exp(-(F_base[:, None] + lr * pred_block)))
        err = (y[:, None] - P) ** 2
        if sample_weight is None:
            return np.mean(err, axis=0)
        w = sample_weight / (np.sum(sample_weight) + 1e-12)
        return np.sum(err * w[:, None], axis=0)


    def _build_candidate_cache(
        self,
        X_tr,
        residual: np.ndarray,
        *,
        pair_meta: Optional[List[Tuple[int, int]]] = None,
        sample_weight=None
    ):
        """
        Addestra (o ri-addestra) tutti i candidati (1D + 2D) sul residuo fornito e,
        se possibile, cacha le predizioni su train.
        """
        # --- 0) Garantisci array NumPy contiguo e dtype prevedibile (senza pandas) ---
        if hasattr(X_tr, "to_numpy"):
            X_tr = X_tr.to_numpy(copy=False)
        X_tr = np.ascontiguousarray(X_tr)
        residual = np.ascontiguousarray(residual)

        n, p = X_tr.shape
        dtype = np.float32

        if pair_meta is not None:
            self._pair_meta = list(pair_meta)
        
        if self.max_pair_candidates is not None and len(self._pair_meta) > self.max_pair_candidates:
            self._pair_meta = self._pair_meta[:self.max_pair_candidates]

        # Utility RAM
        def bytes_cache(shape):
            return int(np.prod(shape)) * np.dtype(dtype).itemsize

        # -------------------------------------------------
        # 1) Univariate learners
        # -------------------------------------------------
        self._uni_trees = []
        if self.group_mode in ("univariate", "mixed"):
            # Maschera boolean per escludere feature già usate / bloccate
            mask_uni = np.ones(p, dtype=bool)
            if getattr(self, "used_features_1d_", None):
                mask_uni[np.fromiter(self.used_features_1d_, dtype=np.intp)] = False
            if getattr(self, "block_univariate_if_in_pair", False) and getattr(self, "_features_used_in_pairs", None):
                mask_uni[np.fromiter(self._features_used_in_pairs, dtype=np.intp)] = False
            valid_uni = np.nonzero(mask_uni)[0]  # array di int

            if self.precompute_univariate and valid_uni.size > 0:
                results = Parallel(n_jobs=-1, prefer="threads")(
                    delayed(self._fit_uni_tree)(
                        i, X_tr, residual, self.max_depth, self.random_state, sample_weight=sample_weight
                    )
                    for i in valid_uni
                )
                trees, preds = zip(*results)  # ciascun preds[k] ha shape (n,)
                self._uni_trees = list(trees)

                # Prealloca piena matrice p-colonne (come nel tuo codice) e riempi solo le valide
                uni_pred = np.zeros((n, p), dtype=dtype)
                # Stack una sola volta (n, n_valid) e assegna a colonne selezionate
                pred_mat = np.stack(preds, axis=1).astype(dtype, copy=False)
                uni_pred[:, valid_uni] = pred_mat
                self._uni_pred_cache = uni_pred
            else:
                self._uni_pred_cache = None
            self._uni_index_map = {int(j): int(k) for k, j in enumerate(valid_uni)}
        else:
            self._uni_pred_cache = None

        # -------------------------------------------------
        # 2) Pairwise learners
        # -------------------------------------------------
        q_total = len(self._pair_meta)

        # Filtra solo le coppie effettivamente candidabili
        if q_total > 0:
            if getattr(self, "allow_pair_reuse", False):
                valid_pairs = list(self._pair_meta)
            else:
                # normalizza le tuple per confronti veloci
                used = getattr(self, "_used_pair_tuples", set())
                valid_pairs = [(i, j) for (i, j) in self._pair_meta
                            if (min(i, j), max(i, j)) not in used]
            q_valid = len(valid_pairs)
        else:
            valid_pairs, q_valid = [], 0

        # Stima memoria solo sui necessari
        need_uni = bytes_cache((n, p)) if self._uni_pred_cache is not None else 0
        need_pair = bytes_cache((n, q_valid)) if (self.precompute_pairs and q_valid > 0) else 0

        use_pair_cache = self.precompute_pairs
        if self.cache_max_mb is not None:
            limit = int(self.cache_max_mb * 1024 * 1024)
            if need_uni + need_pair > limit and use_pair_cache:
                if self.verbose >= 1:
                    logger.info(f"[DICT] Disabilito pair cache (RAM limit). Richiesti {need_uni + need_pair}B, limite {limit}B.")
                need_pair = 0
                use_pair_cache = False
            if need_uni + need_pair > limit and self._uni_pred_cache is not None:
                logger.info("[DICT] Disabilito uni cache (RAM limit).")
                self._uni_pred_cache = None
                need_uni = 0

        if q_valid > 0:
            if use_pair_cache:
                results = Parallel(n_jobs=-1, prefer="threads")(
                    delayed(self._fit_pair_tree)(
                        i, j, X_tr, residual, self.max_depth, self.random_state, sample_weight=sample_weight
                    )
                    for (i, j) in valid_pairs
                )
                trees, preds = zip(*results)
                self._pair_trees_cache = list(trees)  # cache-aligned
                pred_mat = np.stack(preds, axis=1).astype(dtype, copy=False)
                self._pair_pred_cache = pred_mat
            else:
                self._pair_pred_cache = None
                self._pair_trees_cache = None

            # mappa cache-col -> indice in _pair_meta
            idx_map = { (min(i,j), max(i,j)) : k for k,(i,j) in enumerate(self._pair_meta) }
            self._pair_cache_meta_idx = np.array(
                [ idx_map[(min(i,j), max(i,j))] for (i,j) in valid_pairs ],
                dtype=np.int32
            )
        else:
            self._pair_pred_cache = None
            self._pair_trees_cache = None
            self._pair_cache_meta_idx = None


    def _capture_feature_names(self, X):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray([str(c) for c in X.columns])
        elif hasattr(X, "feature_names_in_"):
            self.feature_names_in_ = np.asarray([str(c) for c in X.feature_names_in_])
        else:
            n = int(X.shape[1])
            self.feature_names_in_ = np.asarray([f"f{i}" for i in range(n)])
        self.n_features_in_ = int(X.shape[1])

    def _to_numpy(self, X):
        return X.values if hasattr(X, "values") else np.asarray(X)

    def _raw_margin(self, X: ArrayLike) -> np.ndarray:
        X = check_array(X, dtype=np.float32)
        F = np.full(X.shape[0], self.log_odds_p0_, dtype=np.float32)
        for st in self.stages_:
            F += self.lr * st.tree.predict(X[:, list(st.feats)])
        return F

    def _as_array(self, X):
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
    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: Optional[np.ndarray] = None):
        # ---- logger level coerente con verbose ----
        _prev_level = logger.level
        try:
            if getattr(self, "verbose", 0) >= 2:
                logger.setLevel(logging.DEBUG)
            elif getattr(self, "verbose", 0) >= 1:
                logger.setLevel(logging.INFO)
            else:
                logger.setLevel(logging.WARNING)

            # ---- feature names / importance preliminari ----
            self._capture_feature_names(X)
            if self.feature_importance_fn is not None:
                out = self.feature_importance_fn(X, y)
                self.feature_order = out[0] if isinstance(out, (tuple, list)) else out

            X_in, y_in = X, y
            X_arr, colmap = self._as_array(X_in)  # mappatura nome->indice
            X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float32)

            # Se l'utente passa sample_weight, valida solo la shape ORA (prima di eventuali split)
            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
                if sample_weight.shape[0] != y.shape[0]:
                    raise ValueError(
                        f"sample_weight length mismatch: got {sample_weight.shape[0]} vs y={y.shape[0]}"
                    )

            # ---- validazioni ----
            u = np.unique(y)
            if not np.array_equal(u, np.array([0.0, 1.0])):
                raise ValueError("Questo classificatore supporta y binaria {0,1}.")
            if self.selection not in ("greedy", "static"):
                raise ValueError("selection deve essere 'greedy' o 'static'.")

            n, p = X.shape
            if self.verbose >= 1:
                logger.info(
                    f"Start training on {n} samples, {p} features, "
                    f"max {self.n_stages or 'auto'} stages, "
                    f"mode={self.group_mode}, selection={self.selection}"
                )

            # ---- reset stato interno ----
            self.stages_.clear()
            self.used_features_1d_.clear()
            self.used_pairs_.clear()
            self._used_pair_tuples.clear()
            self.validation_scores_.clear()
            self._fast_pairs_.clear()
            self._features_used_in_pairs.clear()
            self.n_estimators_ = 0
            self._pair_cache_meta_idx = None
            self._pair_trees_cache = None

            # ---- split validation (se non fornita) ----
            # Nota: se eseguiamo lo split qui, splittiamo ANCHE i pesi.
            sw_tr = None  # verrà popolato se necessario
            if (self.validation_X is None) or (self.validation_y is None):
                if sample_weight is not None:
                    X, self.validation_X, y, self.validation_y, sw_tr, _sw_val = train_test_split(
                        X, y, sample_weight,
                        test_size=self.validation_fraction,
                        random_state=self.random_state,
                        stratify=y
                    )
                else:
                    X, self.validation_X, y, self.validation_y = train_test_split(
                        X, y,
                        test_size=self.validation_fraction,
                        random_state=self.random_state,
                        stratify=y
                    )
            # else: validation fornita dall'utente -> non tocchiamo X,y,sample_weight

            # ---- early stopping: prepara tr/val ----
            if self.early_stopping:
                if self.validation_X is not None and self.validation_y is not None:
                    X_tr, y_tr = X, y
                    X_val = check_array(self.validation_X, dtype=np.float32)
                    y_val = np.asarray(self.validation_y, dtype=np.float32)
                    # allinea i pesi al train corrente
                    if sample_weight is not None and sw_tr is not None:
                        sample_weight_tr = np.asarray(sw_tr, dtype=np.float32).reshape(-1)
                    else:
                        sample_weight_tr = None
                else:
                    # (caso raro, mantenuto per compatibilità)
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X, y,
                        test_size=self.validation_fraction,
                        random_state=self.random_state,
                        stratify=y
                    )
                    if sample_weight is not None:
                        # se avevamo pesi, rifacciamo lo split in parallelo
                        sample_weight_tr, _sw_val2 = train_test_split(
                            sample_weight,
                            test_size=self.validation_fraction,
                            random_state=self.random_state,
                            stratify=y  # stratify sullo stesso y di sopra
                        )
                    else:
                        sample_weight_tr = None
            else:
                X_tr, y_tr = X, y
                X_val = None
                y_val = None
                sample_weight_tr = np.asarray(sw_tr, dtype=np.float32).reshape(-1) if sw_tr is not None else (
                    np.asarray(sample_weight, dtype=np.float32).reshape(-1) if sample_weight is not None else None
                )

            # ---- auto-balance dei pesi (SOLO se l'utente non li ha forniti) ----
            if sample_weight_tr is None:
                # class_weight="balanced" equivalente sui soli campioni di training
                n_pos = float(np.sum(y_tr == 1))
                n_neg = float(np.sum(y_tr == 0))
                w_pos = (len(y_tr) / (2.0 * (n_pos + 1e-12)))
                w_neg = (len(y_tr) / (2.0 * (n_neg + 1e-12)))
                sample_weight_tr = np.where(y_tr == 1, w_pos, w_neg).astype(np.float32)

            # Safety check: i pesi devono combaciare con y_tr
            if sample_weight_tr.shape[0] != y_tr.shape[0]:
                raise ValueError(
                    f"internal error: sample_weight_tr length {sample_weight_tr.shape[0]} "
                    f"!= y_tr length {y_tr.shape[0]}"
                )

            # D'ora in avanti usa SEMPRE 'sample_weight = sample_weight_tr' per coerenza
            sample_weight = sample_weight_tr

            # ---- inizializza margini ----
            self._init_base_score(y_tr)  # definisce p0_, log_odds_p0_, ecc.
            F_tr = np.full(y_tr.shape[0], self.log_odds_p0_, dtype=np.float32)
            F_val = np.full(y_val.shape[0], self.log_odds_p0_, dtype=np.float32) if self.early_stopping else None

            # ---- early stopping tracking ----
            best_score = None
            best_iter = 0
            prev_score = None
            no_improve_rounds = 0
            worsening_rounds = 0
            es_active = (self.early_stopping and self.group_mode in ("univariate", "pairwise"))

            if self.early_stopping:
                s0 = self._score_from_margin(
                    F_val, y_val,
                    metric=self.early_stopping_metric,
                    thr=self.threshold,
                    eps=self.epsilon
                )
                self.validation_scores_.append(s0)
                best_score = s0
                prev_score = s0
                if self.verbose >= 1:
                    logger.info(f"[Iter 0] val_{self.early_stopping_metric}={s0:.6f}")

            # ---- FAST iniziale ----
            if self.group_mode in ("pairwise", "mixed"):
                residual0 = y_tr - expit(F_tr)
                feat_names = [str(nm) for nm in self.feature_names_in_]
                pairs, weights = FAST.run(
                    X=X_tr,
                    residuals=residual0,
                    feature_names=feat_names,
                    bins=self.fast_bins,
                    num_threads=self.fast_num_threads,
                    min_support=self.fast_min_support,
                    max_cells=self.fast_max_cells,
                )
                order = np.argsort(-np.asarray(weights))
                self._fast_pairs_ = [pairs[k] for k in order]
                self._pair_meta = list(self._fast_pairs_)
                if self.verbose >= 2 and len(self._pair_meta) > 0:
                    logger.debug(f"FAST initialised with {len(self._pair_meta)} candidate pairs")
            else:
                self._fast_pairs_ = []
                self._pair_meta = []

            # ---- Cache iniziale ----
            self._Xtr_cache = X_tr
            r0 = y_tr - self.p0_
            self._build_candidate_cache(X_tr, r0, pair_meta=self._pair_meta, sample_weight=sample_weight)

            # ---- iterazioni massime ----
            if self.n_stages is not None:
                T = int(self.n_stages)
            else:
                q = len(self._pair_meta) if self.group_mode in ("pairwise", "mixed") else 0
                T = int(p + q) if self.group_mode in ("pairwise", "mixed") else int(p)

            # ========== Progress bar per verbose==1 ==========
            pbar = None
            if self.verbose == 1:
                try:
                    pbar = _tqdm(
                        total=T,
                        desc="Training",
                        leave=False,
                        dynamic_ncols=True,
                        mininterval=0.3,   # throttling refresh
                    )
                except Exception:
                    pbar = None  # se tqdm non disponibile, continua senza


            # =================================================
            # FASE 1: univariate (STATIC)
            # =================================================
            if self.selection == "static":
                # costruisci ordine delle feature
                if self.feature_importance_fn is not None:
                    out = self.feature_importance_fn(X_in, y_in)

                    def _map_to_idx(seq_like):
                        idxs, seen = [], set()
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

                    order_candidate = out[0] if (isinstance(out, (tuple, list)) and len(out) >= 1) else out
                    order_idx = None
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
                        raise ValueError("feature_importance_fn non ha restituito un ordine valido.")
                else:
                    order_idx = list(range(p)) if self.feature_order is None else [
                        int(colmap[o]) if o in colmap else int(o) for o in self.feature_order
                    ]

                order_idx = order_idx[:T]

                for t, i in enumerate(order_idx, 1):
                    residual = y_tr - expit(F_tr)
                    tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
                    tree.fit(X_tr[:, [i]], residual, sample_weight=sample_weight if sample_weight is not None else None)

                    pred_tr = tree.predict(X_tr[:, [i]])
                    base_loss_tr = self._loss_scalar(F_tr, y_tr, self.greedy_metric, self.epsilon, sample_weight=sample_weight)
                    loss_after = self._loss_scalar(F_tr + self.lr * pred_tr, y_tr, self.greedy_metric, self.epsilon, sample_weight=sample_weight)
                    F_tr = F_tr + self.lr * pred_tr
                    stage_gain = base_loss_tr - loss_after

                    self.stages_.append(_Stage(feats=(i,), tree=tree, kind="uni", score=float(stage_gain)))
                    self.used_features_1d_.append(i)

                    # iterativo: DEBUG se verbose>=2
                    if self.verbose >= 2:
                        logger.debug(f"[Iter {t}/{T}] kind=uni, features=(f{i}), train_gain={stage_gain:.4f}")

                    # progress bar se verbose==1
                    if pbar is not None:
                        if (t == 1) or (t % 25 == 0):
                            pbar.set_postfix_str(f"uni f{i} gain={stage_gain:.4f}")
                        pbar.update(1)

                    # ---- ES (macro) ----
                    if self.early_stopping and X_val is not None:
                        pred_val = tree.predict(X_val[:, [i]])
                        F_val = F_val + self.lr * pred_val
                        if self.early_stopping_metric.lower() == "f1":
                            p_val = expit(F_val)
                            prec, rec, _thr = precision_recall_curve(y_val, p_val)
                            f1s = 2 * prec * rec / (prec + rec + 1e-12)
                            val_score = float(np.nanmax(f1s))
                        else:
                            val_score = self._score_from_margin(
                                F_val, y_val,
                                metric=self.early_stopping_metric,
                                thr=self.threshold,
                                eps=self.epsilon
                            )
                        self.validation_scores_.append(val_score)

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

                        if no_improve_rounds >= self.patience or worsening_rounds >= self.patience:
                            if self.verbose >= 1:
                                reason = "no_improve" if no_improve_rounds >= self.patience else "worsening"
                                logger.info(
                                    f"Early stopping at iter {t} due to {reason} "
                                    f"(best_iter={best_iter}, best_score={best_score:.6f})"
                                )
                            if pbar is not None:
                                pbar.close()
                            break
                        prev_score = val_score

            # =================================================
            # FASE 2: greedy uni+pair
            # =================================================
            start_t = len(self.stages_) + 1
            for t in range(start_t, T + 1):
                base_loss_tr = self._loss_scalar(F_tr, y_tr, self.greedy_metric, self.epsilon, sample_weight=sample_weight)
                residual = y_tr - expit(F_tr)

                # --- Univariate cand. (cache -> block scoring + refit opzionale) ---
                best_uni_i = None
                best_uni_pred = None
                best_uni_tree = None
                best_uni_loss = np.inf

                if self.group_mode in ("univariate", "mixed"):
                    if self._uni_pred_cache is not None:
                        mask_uni = np.ones(p, dtype=bool)
                        if self.used_features_1d_:
                            mask_uni[np.fromiter(self.used_features_1d_, dtype=np.intp)] = False
                        if self.block_univariate_if_in_pair and self._features_used_in_pairs:
                            mask_uni[np.fromiter(self._features_used_in_pairs, dtype=np.intp)] = False
                        valid_idx = np.nonzero(mask_uni)[0]

                        if valid_idx.size > 0:
                            score_block = self._logloss_block if self.greedy_metric == "logloss" else self._mse_block
                            if self.greedy_metric == "logloss":
                                L = score_block(F_tr, y_tr, self._uni_pred_cache[:, valid_idx], self.lr, self.epsilon, sample_weight=sample_weight)
                            else:
                                L = score_block(F_tr, y_tr, self._uni_pred_cache[:, valid_idx], self.lr, sample_weight=sample_weight)

                            k_rel = int(np.argmin(L))
                            best_uni_i = int(valid_idx[k_rel])
                            best_uni_loss = float(L[k_rel])

                            # prendi l'albero dalla cache; rifitta SOLO se vuoi massima aderenza al residuo
                            k_tree = self._uni_index_map.get(best_uni_i, None)
                            if k_tree is not None and k_tree < len(self._uni_trees):
                                cached_tree = self._uni_trees[k_tree]
                            else:
                                cached_tree = None

                            # opzionale ma consigliato: rifitta SOLO il vincitore sul residuo corrente
                            if cached_tree is None:
                                t_uni = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
                                t_uni.fit(X_tr[:, [best_uni_i]], residual, sample_weight=sample_weight if sample_weight is not None else None)
                                best_uni_tree = t_uni
                                best_uni_pred = t_uni.predict(X_tr[:, [best_uni_i]])
                            else:
                                # usa l'albero cached; se vuoi, rifittalo per coerenza 100%
                                # (commenta le 3 righe seguenti se preferisci non rifittare)
                                t_uni = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
                                t_uni.fit(X_tr[:, [best_uni_i]], residual, sample_weight=sample_weight if sample_weight is not None else None)
                                cached_tree = t_uni
                                self._uni_pred_cache[:, best_uni_i] = t_uni.predict(X_tr[:, [best_uni_i]]).astype(np.float32)

                                best_uni_tree = cached_tree
                                best_uni_pred = self._uni_pred_cache[:, best_uni_i].astype(np.float32)

                    else:
                        # Fallback senza cache: (meno efficiente) – rimane la tua versione per compatibilità
                        for i in range(p):
                            if i in self.used_features_1d_:
                                continue
                            if self.block_univariate_if_in_pair and (i in self._features_used_in_pairs):
                                continue
                            tree_i = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
                            tree_i.fit(X_tr[:, [i]], residual)
                            pred_i = tree_i.predict(X_tr[:, [i]])
                            loss_i = self._loss_scalar(F_tr + self.lr * pred_i, y_tr, self.greedy_metric, self.epsilon, sample_weight=sample_weight)
                            if loss_i < best_uni_loss:
                                best_uni_loss = float(loss_i)
                                best_uni_i = i
                                best_uni_pred = pred_i
                                best_uni_tree = tree_i


                # --- Pairwise cand. (cache -> refit) ---
                best_pair_idx = None
                best_pair_feats = None
                best_pair_loss_cached = np.inf
                if self.group_mode in ("pairwise", "mixed") and len(self._pair_meta) > 0:
                    if self._pair_pred_cache is not None:
                        valid_cols = np.arange(self._pair_pred_cache.shape[1], dtype=int)

                        if not self.allow_pair_reuse and len(self._used_pair_tuples) > 0:
                            mask = np.ones(valid_cols.size, dtype=bool)
                            for k, col in enumerate(valid_cols):
                                meta_idx = int(self._pair_cache_meta_idx[col])     # mappa colonna cache -> indice in _pair_meta
                                ii, jj = self._pair_meta[meta_idx]
                                if (min(ii, jj), max(ii, jj)) in self._used_pair_tuples:
                                    mask[k] = False
                            valid_cols = valid_cols[mask]

                        if valid_cols.size > 0:
                            score_block = self._logloss_block if self.greedy_metric == "logloss" else self._mse_block
                            for s in range(0, valid_cols.size, self.score_batch):
                                cols = valid_cols[s:s + self.score_batch]
                                if self.greedy_metric == "logloss":
                                    L = score_block(F_tr, y_tr, self._pair_pred_cache[:, cols], self.lr, self.epsilon, sample_weight=sample_weight)
                                else:
                                    L = score_block(F_tr, y_tr, self._pair_pred_cache[:, cols], self.lr, sample_weight=sample_weight)

                                k_rel = int(np.argmin(L))
                                if L[k_rel] < best_pair_loss_cached:
                                    best_pair_loss_cached = float(L[k_rel])
                                    best_pair_idx = int(cols[k_rel])                  # <-- questo è l’INDICE DI COLONNA della cache
                                    meta_idx = int(self._pair_cache_meta_idx[best_pair_idx])
                                    best_pair_feats = self._pair_meta[meta_idx]       # <-- ecco la coppia giusta

                    else:
                        for k_idx, (ii, jj) in enumerate(self._pair_meta):
                            if (not self.allow_pair_reuse) and ((min(ii, jj), max(ii, jj)) in self._used_pair_tuples):
                                continue
                            pred = (
                                DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
                                .fit(X_tr[:, [ii, jj]], residual)  # <-- residuo CORRENTE, non y_tr - self.p0_
                                .predict(X_tr[:, [ii, jj]])
                                .astype(np.float32)
                            )

                            loss_c = self._loss_scalar(F_tr + self.lr * pred, y_tr, self.greedy_metric, self.epsilon, sample_weight=sample_weight)
                            if loss_c < best_pair_loss_cached:
                                best_pair_loss_cached = float(loss_c)
                                best_pair_idx = k_idx
                                best_pair_feats = (ii, jj)

                best_pair_tree = None
                best_pair_pred_refit = None
                best_pair_loss_refit = np.inf
                if best_pair_idx is not None and best_pair_feats is not None:
                    ii, jj = best_pair_feats
                    tree_pair = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
                    tree_pair.fit(X_tr[:, [ii, jj]], residual, sample_weight=sample_weight if sample_weight is not None else None)
                    pred_pair = tree_pair.predict(X_tr[:, [ii, jj]]).astype(np.float32)
                    loss_pair = self._loss_scalar(F_tr + self.lr * pred_pair, y_tr, self.greedy_metric, self.epsilon, sample_weight=sample_weight)
                    best_pair_tree = tree_pair
                    best_pair_pred_refit = pred_pair
                    best_pair_loss_refit = float(loss_pair)
                    if self._pair_pred_cache is not None:
                        self._pair_pred_cache[:, best_pair_idx] = pred_pair.astype(np.float32)
                    if getattr(self, "_pair_trees_cache", None) is not None:
                        self._pair_trees_cache[best_pair_idx] = tree_pair


                # --- scelta candidato ---
                candidates = []
                if best_uni_i is not None:
                    candidates.append(("uni", best_uni_loss, (best_uni_i,), best_uni_tree, best_uni_pred))
                if best_pair_tree is not None:
                    candidates.append(("pair", best_pair_loss_refit, tuple(best_pair_feats), best_pair_tree, best_pair_pred_refit))
                if not candidates:
                    if pbar is not None:
                        pbar.close()
                    break

                candidates.sort(key=lambda x: x[1])
                chosen_kind, loss_after, feats, tree, pred_tr = candidates[0]
                # se i due migliori sono vicinissimi, rifitta e confronta di nuovo
                if len(candidates) > 1 and (candidates[1][1] - candidates[0][1]) < 1e-4:
                    pass  # già rifitti i vincitori; opzionale rifit anche del secondo se non lo era

                # bookkeeping
                if chosen_kind == "uni":
                    self.used_features_1d_.append(int(feats[0]))
                else:
                    ii, jj = int(feats[0]), int(feats[1])
                    if not self.allow_pair_reuse:
                        self._used_pair_tuples.add((min(ii, jj), max(ii, jj)))
                    if self.block_univariate_if_in_pair:
                        self._features_used_in_pairs.update((ii, jj))
                    es_active = True
                
                # ---- Refresh periodici (ADATTIVO) ----
                need_cache_refresh = (chosen_kind == "pair") or (t % self.cache_refresh_every == 0)
                if need_cache_refresh:
                    if self.verbose >= 2:
                        logger.debug(f"[Iter {t}] Refresh candidate cache (adaptive)")
                    self._build_candidate_cache(X_tr, y_tr - expit(F_tr), pair_meta=self._pair_meta, sample_weight=sample_weight)

                if self.group_mode in ("pairwise", "mixed") and self.fast_refresh_every and (t % self.fast_refresh_every == 0):
                    if self.verbose >= 2:
                        logger.debug(f"[Iter {t}] Refresh FAST")
                    residual_t = y_tr - expit(F_tr)
                    feat_names = [str(nm) for nm in self.feature_names_in_]
                    new_pairs, new_w = FAST.run(
                        X=X_tr, residuals=residual_t, feature_names=feat_names,
                        bins=self.fast_bins, num_threads=self.fast_num_threads,
                        min_support=self.fast_min_support, max_cells=self.fast_max_cells,
                    )
                    order = np.argsort(-np.asarray(new_w))
                    new_pairs = [new_pairs[k] for k in order]
                    existing = set((min(i, j), max(i, j)) for (i, j) in self._pair_meta)
                    added = 0
                    for (i, j) in new_pairs:
                        key = (min(i, j), max(i, j))
                        if key not in existing:
                            self._pair_meta.append((i, j))
                            existing.add(key)
                            added += 1
                    if self.verbose >= 2 and added > 0:
                        logger.debug(f"[Iter {t}] FAST added {added} new pairs")
                    if added > 0:
                        self._build_candidate_cache(X_tr, y_tr - expit(F_tr), pair_meta=self._pair_meta, sample_weight=sample_weight)

                # update stato
                F_tr = F_tr + self.lr * pred_tr
                stage_gain = base_loss_tr - float(loss_after)
                self.stages_.append(_Stage(feats=feats, tree=tree, kind=chosen_kind, score=stage_gain))

                # iterativo: DEBUG se verbose>=2
                if self.verbose >= 2:
                    logger.debug(f"[Iter {t}/{T}] kind={chosen_kind}, features={feats}, train_gain={stage_gain:.4f}")

                # progress bar se verbose==1
                if pbar is not None:
                    if (t == start_t) or (t % 25 == 0):
                        if chosen_kind == "uni":
                            pbar.set_postfix_str(f"uni f{feats[0]} gain={stage_gain:.4f}")
                        else:
                            pbar.set_postfix_str(f"pair ({feats[0]},{feats[1]}) gain={stage_gain:.4f}")
                    pbar.update(1)

                # ---- ES (macro) ----
                if self.early_stopping and X_val is not None:
                    pred_val = tree.predict(X_val[:, list(feats)]).astype(np.float32)
                    F_val = F_val + self.lr * pred_val
                    if self.early_stopping_metric.lower() == "f1":
                        p_val = expit(F_val)
                        prec, rec, _thr = precision_recall_curve(y_val, p_val)
                        f1s = 2 * prec * rec / (prec + rec + 1e-12)
                        val_score = float(np.nanmax(f1s))
                    else:
                        val_score = self._score_from_margin(
                            F_val, y_val,
                            metric=self.early_stopping_metric,
                            thr=self.threshold,
                            eps=self.epsilon
                        )
                    self.validation_scores_.append(val_score)

                    improved = (best_score is None) or (val_score > best_score + self.tol)
                    worsened = (prev_score is not None) and (val_score < prev_score - self.tol)
                    if improved:
                        best_score = val_score
                        best_iter = t
                        if es_active:
                            no_improve_rounds = 0
                    else:
                        if es_active:
                            no_improve_rounds += 1
                    if worsened and es_active:
                        worsening_rounds += 1
                    else:
                        worsening_rounds = 0

                    if es_active and (no_improve_rounds >= self.patience or worsening_rounds >= self.patience):
                        if self.verbose >= 1:
                            reason = "no_improve" if no_improve_rounds >= self.patience else "worsening"
                            logger.info(
                                f"Early stopping at iter {t} due to {reason} "
                                f"(best_iter={best_iter}, best_score={best_score:.6f})"
                            )
                        if pbar is not None:
                            pbar.close()
                        break
                    prev_score = val_score

            # ---- truncation a best_iter (macro) ----
            if self.early_stopping and best_iter < len(self.stages_):
                self.stages_ = self.stages_[:best_iter]
                self.best_iteration_ = best_iter
                self.best_score_ = best_score
                if self.verbose >= 1:
                    logger.info(f"Final model truncated at iter {best_iter} (best_score={best_score:.6f})")

            # ---- POST-PRUNING (macro) ----
            if self.group_mode == "mixed":
                pair_scores = [st.score for st in self.stages_ if st.kind == "pair"]
                if len(pair_scores) > 0:
                    thr = float(np.min(pair_scores))
                    kept = []
                    for st in self.stages_:
                        if (st.kind == "pair") or (st.score >= thr):
                            kept.append(st)
                    drop_cnt = len(self.stages_) - len(kept)
                    if self.verbose >= 1 and drop_cnt > 0:
                        logger.info(f"Pruned {drop_cnt} univariates with score < {thr:.6g}")
                    self.stages_ = kept
                    self.used_features_1d_ = sorted({i for st in self.stages_ if st.kind == "uni" for i in st.feats})

            # ---- ricostruisci used_pairs_ ----
            self.used_pairs_.clear()
            for st in self.stages_:
                if st.kind == "pair":
                    try:
                        idx = self._pair_meta.index((st.feats[0], st.feats[1]))
                    except ValueError:
                        try:
                            idx = self._pair_meta.index((st.feats[1], st.feats[0]))
                        except ValueError:
                            continue
                    self.used_pairs_.append(idx)

            # chiudi eventuale pbar rimasta aperta
            if pbar is not None:
                pbar.close()

            self.n_estimators_ = len(self.stages_)
            return self

        finally:
            logger.setLevel(_prev_level)


    # ---------- inference ----------
    def decision_function(self, X: ArrayLike) -> np.ndarray:
        return self._raw_margin(X)

    def predict_proba(self, X):
        z = self.decision_function(X)
        p1 = _sigmoid(z)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: ArrayLike) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


    # ----- utilities -----

    def get_used_features(self) -> List:
        used = set()
        for st in self.stages_:
            for i in st.feats:
                used.add(i)
        if self.feature_names_in_ is None:
            return sorted(list(used))
        return [self.feature_names_in_[i] for i in sorted(list(used))]

    def get_stages(self) -> List[dict]:
        out = []
        for k, st in enumerate(self.stages_, 1):
            names = [self.feature_names_in_[i] if self.feature_names_in_ is not None else i for i in st.feats]
            out.append({
                "iter": k,
                "kind": st.kind,
                "features": tuple(names),
                "max_depth": self.max_depth
            })
        return out

    def get_early_stopping_info(self) -> dict:
        return {
            "early_stopping_used": self.early_stopping,
            "best_iteration": self.best_iteration_ if self.early_stopping else None,
            "best_score": self.best_score_ if self.early_stopping else None,
            "final_estimators": self.n_estimators_,
            "validation_scores": self.validation_scores_ if self.early_stopping else [],
            "early_stopping_metric": self.early_stopping_metric if self.early_stopping else None,
        }

    def _fit_uni_tree(self, i, X, residual, max_depth, random_state, sample_weight=None):
        Xi = np.ascontiguousarray(X[:, i:i+1])
        t = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        t.fit(Xi, residual, sample_weight=sample_weight)
        return t, t.predict(Xi)

    def _fit_pair_tree(self, i, j, X, residual, max_depth, random_state, sample_weight=None):
        Xij = np.ascontiguousarray(X[:, (i, j)])
        t = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        t.fit(Xij, residual, sample_weight=sample_weight)
        return t, t.predict(Xij)

import time
import numpy as np
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score,
    matthews_corrcoef, accuracy_score
)
from scipy.stats import wilcoxon
from utils import get_available_datasets, load_preprocess, get_feature_importance

all_data = get_available_datasets()
datasets = []
for name, info in all_data.items():
    _data = load_preprocess(name)
    datasets.append({"name": name, "X": _data["X"], "y": _data["y"]})

BASE_PARAMS = dict(
    max_depth=2,
    lr=1.0,
    selection="greedy",
    early_stopping=True,
    early_stopping_metric="auc",
    validation_fraction=0.2,
    patience=5,
    tol=1e-4,
    random_state=42,
    epsilon=1e-12,
    verbose=0,
    fast_bins=8,
    cache_refresh_every=1,
    fast_refresh_every=5
)

PARAMS_UNIVARIATE = dict(**BASE_PARAMS, group_mode="univariate")
PARAMS_PAIRWISE   = dict(**BASE_PARAMS, group_mode="pairwise",
                         allow_pair_reuse=False)
PARAMS_MIXED      = dict(**BASE_PARAMS, group_mode="mixed",
                         allow_pair_reuse=False, block_univariate_if_in_pair=True)

MODEL_PARAMS = {
    "univariate": PARAMS_UNIVARIATE,
    "pairwise":   PARAMS_PAIRWISE,
    "mixed":      PARAMS_MIXED,
}

PAIRS = [
    ("pairwise", "univariate"),
    ("mixed", "univariate"),
    ("mixed", "pairwise"),
]

METRICS = ["auc", "f1", "balanced_accuracy", "mcc", "accuracy"]

def _ensure_binary_numpy_y(y):
    y_np = y.values if hasattr(y, "values") else np.asarray(y)
    uniq = np.unique(y_np)
    if set(uniq.tolist()) == {0, 1}:
        return y_np.astype(int)
    if len(uniq) != 2:
        raise ValueError(f"y non binaria (trovate {len(uniq)} classi): {uniq}")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y_np)
    if set(np.unique(y_enc)) != {0, 1}:
        y_enc = (y_enc == y_enc.max()).astype(int)
    return y_enc

def _split_index(X, y, tr_idx, te_idx):
    if hasattr(X, "iloc"):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    else:
        Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]
    return Xtr, Xte, ytr, yte

def _compute_metrics(y_true, proba, pred):
    if np.unique(y_true).size < 2:
        return None
    return dict(
        auc=roc_auc_score(y_true, proba),
        f1=f1_score(y_true, pred, zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y_true, pred),
        mcc=matthews_corrcoef(y_true, pred),
        accuracy=accuracy_score(y_true, pred),
    )

def eval_dataset_metrics_mean_multi(
    X, y, model_params: Dict[str, dict],
    cv_splits: int = 5, seed: int = 123,
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, float]:
    """
    Returns a dict with (for each metric m and model k):
      m_{k}  + times train_time_{k}_s  + (opt.) delta_{m}_{B}_minus_{A}
    and optional speed_{B}_vs_{A} = time_A / time_B ( >1 => B faster than A ).
    """
    X_in = X
    y_np = _ensure_binary_numpy_y(y)
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

    model_names = list(model_params.keys())
    folds = {name: {m: [] for m in METRICS} for name in model_names}
    times = {name: 0.0 for name in model_names}

    for tr, te in tqdm(skf.split(X_in if not hasattr(X_in, "values") else X_in.values, y_np)):
        Xtr, Xte, ytr, yte = _split_index(X_in, y_np, tr, te)
        if np.unique(yte).size < 2:
            continue

        for name in model_names:
            clf = Model2D(**model_params[name])
            t0 = time.perf_counter()
            clf.fit(Xtr, ytr)
            times[name] += time.perf_counter() - t0
            proba = clf.predict_proba(Xte)[:, 1]
            pred = clf.predict(Xte)
            res = _compute_metrics(yte, proba, pred)
            if res is None:
                continue
            for m in METRICS:
                folds[name][m].append(res[m])

    if any(len(folds[model_names[0]][m]) == 0 for m in METRICS):
        raise RuntimeError("CV degenerata: nessun fold valido per una o più metriche.")

    out = {}
    for name in model_names:
        for m in METRICS:
            out[f"{m}_{name}"] = float(np.mean(folds[name][m]))
        out[f"train_time_{name}_s"] = float(times[name])

    if pairs:
        for B, A in pairs:
            for m in METRICS:
                out[f"delta_{m}_{B}_minus_{A}"] = out[f"{m}_{B}"] - out[f"{m}_{A}"]
            tA, tB = out[f"train_time_{A}_s"], out[f"train_time_{B}_s"]
            out[f"speed_{B}_vs_{A}"] = (tA / tB) if tB > 0 else np.nan

    return out

import traceback
def run_wilcoxon_over_datasets_multi3(
    datasets: List[Dict[str, object]],
    model_params: Dict[str, dict],
    cv_splits: int = 5,
    seed: int = 123,
    pairs: Optional[List[Tuple[str, str]]] = None,
):
    rows = []
    model_names = list(model_params.keys())

    for ds in datasets:
        print(f"\n=========== Dataset: {ds['name']} ===========")
        name, X, Y = ds["name"], ds["X"], ds["y"]
        try:
            r = eval_dataset_metrics_mean_multi(
                X, Y, model_params=model_params,
                cv_splits=cv_splits, seed=seed, pairs=pairs
            )
            rows.append({"dataset": name, **r})
        except Exception as e:
            print(f"\n[ERRORE su dataset '{name}'] {type(e).__name__}: {e}")
            traceback.print_exc()

            nan_row = {"dataset": name}
            for k in model_names:
                for m in METRICS:
                    nan_row[f"{m}_{k}"] = np.nan
                nan_row[f"train_time_{k}_s"] = np.nan
            if pairs:
                for B, A in pairs:
                    for m in METRICS:
                        nan_row[f"delta_{m}_{B}_minus_{A}"] = np.nan
                    nan_row[f"speed_{B}_vs_{A}"] = np.nan
            nan_row["note"] = f"{type(e).__name__}: {e}"
            rows.append(nan_row)

    df = pd.DataFrame(rows)

    show_cols = ["dataset"]
    for m in METRICS:
        show_cols += [f"{m}_{k}" for k in model_names]
    if pairs:
        for B, A in pairs:
            show_cols += [f"delta_{m}_{B}_minus_{A}" for m in METRICS]
    time_cols = [f"train_time_{k}_s" for k in model_names]
    speed_cols = [f"speed_{B}_vs_{A}" for (B, A) in (pairs or [])]

    print("\nRAW RESULTS (first columns):")
    print(df[show_cols + time_cols + speed_cols]
          .fillna("").to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    stats = {m: {} for m in METRICS}
    if pairs:
        for m in METRICS:
            for (B, A) in pairs:
                colB, colA = f"{m}_{B}", f"{m}_{A}"
                ok = df.dropna(subset=[colB, colA]).copy()
                if ok.shape[0] < 2:
                    print(f"\n[WARNING] Not enough datasets for Wilcoxon on {m} ({B} vs {A}).")
                    continue
                stat_g, p_g = wilcoxon(ok[colB], ok[colA], zero_method="pratt", alternative="greater")
                stat_2, p_2 = wilcoxon(ok[colB], ok[colA], zero_method="pratt", alternative="two-sided")

                deltas = ok[colB].to_numpy() - ok[colA].to_numpy()
                wins = int(np.sum(deltas > 0))
                ties = int(np.sum(deltas == 0))
                losses = int(np.sum(deltas < 0))
                psup = (wins + 0.5 * ties) / deltas.size
                median_delta = float(np.median(deltas))
                mean_delta = float(np.mean(deltas))

                print(f"\n=== WILCOXON su {m.upper()}  ({B} > {A})  n={ok.shape[0]} ===")
                print(f"One-sided: p = {p_g:.6g}  (W = {stat_g:.3f})")
                print(f"Two-sided: p = {p_2:.6g}")
                print(f"V/P/S: {wins}/{ties}/{losses} | psup ≈ {psup:.3f} | Δ{m} medio={mean_delta:.4f}, mediano={median_delta:.4f}")

                stats[m][(B, A)] = dict(
                    n=int(ok.shape[0]), W=float(stat_g),
                    p_one_sided=float(p_g), p_two_sided=float(p_2),
                    wins=wins, ties=ties, losses=losses,
                    psup=float(psup), delta_mean=mean_delta, delta_median=median_delta
                )

    return df, stats

df_all, stats_all = run_wilcoxon_over_datasets_multi3(
    datasets,
    model_params=MODEL_PARAMS,
    cv_splits=3, seed=123,
    pairs=PAIRS
) # circa 23 minuti
print("\nGLOBAL STATISTICS:")
for m, d in stats_all.items():
    for (B, A), s in d.items():
        print(f"{m} | {B} vs {A} -> {s}")

df_global_stats = pd.DataFrame.from_records(
    [
        {"metric": m, "model": B, "vs": A, **s}
        for m, d in stats_all.items()
        for (B, A), s in d.items()
    ]
)
df_global_stats.to_csv("05_global_stats.csv", index=False)

BASE_PARAMS_STATIC = dict(**BASE_PARAMS)
BASE_PARAMS_STATIC.update(
    selection="static",
    feature_importance_fn=get_feature_importance,
)

PARAMS_UNIVARIATE_STATIC = dict(**BASE_PARAMS_STATIC, group_mode="univariate")
PARAMS_PAIRWISE_STATIC   = dict(**BASE_PARAMS_STATIC, group_mode="pairwise",
                                allow_pair_reuse=False)
PARAMS_MIXED_STATIC      = dict(**BASE_PARAMS_STATIC, group_mode="mixed",
                                allow_pair_reuse=False, block_univariate_if_in_pair=True)

MODEL_PARAMS_STATIC = {
    "univariate": PARAMS_UNIVARIATE_STATIC,
    "pairwise":   PARAMS_PAIRWISE_STATIC,
    "mixed":      PARAMS_MIXED_STATIC,
}

print("\n\n########### RUN: selection=static ###########")
df_all_static, stats_all_static = run_wilcoxon_over_datasets_multi3(
    datasets,
    model_params=MODEL_PARAMS_STATIC,
    cv_splits=3, seed=123,
    pairs=PAIRS
) # circa 15 minuti
df_global_stats_static = pd.DataFrame.from_records(
    [
        {"metric": m, "model": B, "vs": A, **s}
        for m, d in stats_all_static.items()
        for (B, A), s in d.items()
    ]
)
df_global_stats_static.to_csv("05_global_stats_static.csv", index=False)

print("\nGLOBAL STATISTICS (selection=static):")
for m, d in stats_all_static.items():
    for (B, A), s in d.items():
        print(f"{m} | {B} vs {A} -> {s}")

try:
    df_join = df_all_static.merge(df_all, on="dataset", suffixes=("_static", "_greedy"))

    for m in METRICS:
        for k in MODEL_PARAMS_STATIC.keys():
            df_join[f"delta_{m}_{k}_static_minus_greedy"] = (
                df_join[f"{m}_{k}_static"] - df_join[f"{m}_{k}_greedy"]
            )

    cols = ["dataset"]
    for k in MODEL_PARAMS_STATIC.keys():
        cols += [f"auc_{k}_static", f"auc_{k}_greedy", f"delta_auc_{k}_static_minus_greedy"]
    print("\nCONFRONTO STATIC vs GREEDY (AUC):")
    print(df_join[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

except NameError:
    pass

def save_markdown(df: pd.DataFrame, stats: Dict, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Risultati Esperimenti\n\n")

        f.write("## Risultati grezzi\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")

        f.write("## Statistiche Wilcoxon\n\n")
        for m, d in stats.items():
            f.write(f"### Metrica: {m}\n\n")
            for (B, A), s in d.items():
                f.write(f"- **{B} vs {A}**  (n={s['n']})\n")
                f.write(f"  - p (one-sided) = {s['p_one_sided']:.6g}\n")
                f.write(f"  - p (two-sided) = {s['p_two_sided']:.6g}\n")
                f.write(f"  - W = {s['W']:.3f}\n")
                f.write(f"  - Vittorie/Tie/Sconfitte = {s['wins']}/{s['ties']}/{s['losses']}\n")
                f.write(f"  - psup ≈ {s['psup']:.3f}\n")
                f.write(f"  - Δ medio = {s['delta_mean']:.4f}, Δ mediano = {s['delta_median']:.4f}\n\n")

save_markdown(df_all, stats_all, "05_results_greedy.md")
save_markdown(df_all_static, stats_all_static, "05_results_static.md")
if "df_join" in locals():
    with open("05_confronto_static_vs_greedy.md", "w", encoding="utf-8") as f:
        f.write("# Confronto Static vs Greedy\n\n")
        cols = ["dataset"]
        for k in MODEL_PARAMS_STATIC.keys():
            cols += [f"auc_{k}_static", f"auc_{k}_greedy", f"delta_auc_{k}_static_minus_greedy"]
        f.write(df_join[cols].to_markdown(index=False, floatfmt=".4f"))
