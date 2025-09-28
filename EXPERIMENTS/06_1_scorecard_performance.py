from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass
from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple, Union, List, Sequence

from utils import load_preprocess, get_feature_importance, get_available_datasets

import ast
import time

from FAST import FAST
from tqdm import tqdm

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
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

import numpy as np
from itertools import product

# ================ MODEL =======================
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
            return -log_loss(y, p)
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

        def bytes_cache(shape):
            return int(np.prod(shape)) * np.dtype(dtype).itemsize

        self._uni_trees = []
        if self.group_mode in ("univariate", "mixed"):
            mask_uni = np.ones(p, dtype=bool)
            if getattr(self, "used_features_1d_", None):
                mask_uni[np.fromiter(self.used_features_1d_, dtype=np.intp)] = False
            if getattr(self, "block_univariate_if_in_pair", False) and getattr(self, "_features_used_in_pairs", None):
                mask_uni[np.fromiter(self._features_used_in_pairs, dtype=np.intp)] = False
            valid_uni = np.nonzero(mask_uni)[0]

            if self.precompute_univariate and valid_uni.size > 0:
                results = Parallel(n_jobs=-1, prefer="threads")(
                    delayed(self._fit_uni_tree)(
                        i, X_tr, residual, self.max_depth, self.random_state, sample_weight=sample_weight
                    )
                    for i in valid_uni
                )
                trees, preds = zip(*results)
                self._uni_trees = list(trees)

                uni_pred = np.zeros((n, p), dtype=dtype)
                pred_mat = np.stack(preds, axis=1).astype(dtype, copy=False)
                uni_pred[:, valid_uni] = pred_mat
                self._uni_pred_cache = uni_pred
            else:
                self._uni_pred_cache = None
            self._uni_index_map = {int(j): int(k) for k, j in enumerate(valid_uni)}
        else:
            self._uni_pred_cache = None
        q_total = len(self._pair_meta)

        if q_total > 0:
            if getattr(self, "allow_pair_reuse", False):
                valid_pairs = list(self._pair_meta)
            else:
                used = getattr(self, "_used_pair_tuples", set())
                valid_pairs = [(i, j) for (i, j) in self._pair_meta
                            if (min(i, j), max(i, j)) not in used]
            q_valid = len(valid_pairs)
        else:
            valid_pairs, q_valid = [], 0

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
                self._pair_trees_cache = list(trees)
                pred_mat = np.stack(preds, axis=1).astype(dtype, copy=False)
                self._pair_pred_cache = pred_mat
            else:
                self._pair_pred_cache = None
                self._pair_trees_cache = None

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

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: Optional[np.ndarray] = None):
        _prev_level = logger.level
        try:
            if getattr(self, "verbose", 0) >= 2:
                logger.setLevel(logging.DEBUG)
            elif getattr(self, "verbose", 0) >= 1:
                logger.setLevel(logging.INFO)
            else:
                logger.setLevel(logging.WARNING)

            self._capture_feature_names(X)
            if self.feature_importance_fn is not None:
                out = self.feature_importance_fn(X, y)
                self.feature_order = out[0] if isinstance(out, (tuple, list)) else out

            X_in, y_in = X, y
            X_arr, colmap = self._as_array(X_in)
            X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float32)

            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight, dtype=np.float32).reshape(-1)
                if sample_weight.shape[0] != y.shape[0]:
                    raise ValueError(
                        f"sample_weight length mismatch: got {sample_weight.shape[0]} vs y={y.shape[0]}"
                    )

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

            
            sw_tr = None
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
            
            if self.early_stopping:
                if self.validation_X is not None and self.validation_y is not None:
                    X_tr, y_tr = X, y
                    X_val = check_array(self.validation_X, dtype=np.float32)
                    y_val = np.asarray(self.validation_y, dtype=np.float32)

                    if sample_weight is not None and sw_tr is not None:
                        sample_weight_tr = np.asarray(sw_tr, dtype=np.float32).reshape(-1)
                    else:
                        sample_weight_tr = None
                else:
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X, y,
                        test_size=self.validation_fraction,
                        random_state=self.random_state,
                        stratify=y
                    )
                    if sample_weight is not None:
                        sample_weight_tr, _sw_val2 = train_test_split(
                            sample_weight,
                            test_size=self.validation_fraction,
                            random_state=self.random_state,
                            stratify=y
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

            if sample_weight_tr is None:
                n_pos = float(np.sum(y_tr == 1))
                n_neg = float(np.sum(y_tr == 0))
                w_pos = (len(y_tr) / (2.0 * (n_pos + 1e-12)))
                w_neg = (len(y_tr) / (2.0 * (n_neg + 1e-12)))
                sample_weight_tr = np.where(y_tr == 1, w_pos, w_neg).astype(np.float32)

            if sample_weight_tr.shape[0] != y_tr.shape[0]:
                raise ValueError(
                    f"internal error: sample_weight_tr length {sample_weight_tr.shape[0]} "
                    f"!= y_tr length {y_tr.shape[0]}"
                )

            sample_weight = sample_weight_tr

            self._init_base_score(y_tr)
            F_tr = np.full(y_tr.shape[0], self.log_odds_p0_, dtype=np.float32)
            F_val = np.full(y_val.shape[0], self.log_odds_p0_, dtype=np.float32) if self.early_stopping else None

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

            self._Xtr_cache = X_tr
            r0 = y_tr - self.p0_
            self._build_candidate_cache(X_tr, r0, pair_meta=self._pair_meta, sample_weight=sample_weight)

            if self.n_stages is not None:
                T = int(self.n_stages)
            else:
                q = len(self._pair_meta) if self.group_mode in ("pairwise", "mixed") else 0
                T = int(p + q) if self.group_mode in ("pairwise", "mixed") else int(p)

            pbar = None
            if self.verbose == 1:
                try:
                    pbar = tqdm(
                        total=T,
                        desc="Training",
                        leave=False,
                        dynamic_ncols=True,
                        mininterval=0.3,
                    )
                except Exception:
                    pbar = None

            if self.selection == "static":
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

                    if self.verbose >= 2:
                        logger.debug(f"[Iter {t}/{T}] kind=uni, features=(f{i}), train_gain={stage_gain:.4f}")

                    if pbar is not None:
                        if (t == 1) or (t % 25 == 0):
                            pbar.set_postfix_str(f"uni f{i} gain={stage_gain:.4f}")
                        pbar.update(1)

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

            start_t = len(self.stages_) + 1
            for t in range(start_t, T + 1):
                base_loss_tr = self._loss_scalar(F_tr, y_tr, self.greedy_metric, self.epsilon, sample_weight=sample_weight)
                residual = y_tr - expit(F_tr)

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

                            k_tree = self._uni_index_map.get(best_uni_i, None)
                            if k_tree is not None and k_tree < len(self._uni_trees):
                                cached_tree = self._uni_trees[k_tree]
                            else:
                                cached_tree = None

                            if cached_tree is None:
                                t_uni = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
                                t_uni.fit(X_tr[:, [best_uni_i]], residual, sample_weight=sample_weight if sample_weight is not None else None)
                                best_uni_tree = t_uni
                                best_uni_pred = t_uni.predict(X_tr[:, [best_uni_i]])
                            else:
                                t_uni = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
                                t_uni.fit(X_tr[:, [best_uni_i]], residual, sample_weight=sample_weight if sample_weight is not None else None)
                                cached_tree = t_uni
                                self._uni_pred_cache[:, best_uni_i] = t_uni.predict(X_tr[:, [best_uni_i]]).astype(np.float32)

                                best_uni_tree = cached_tree
                                best_uni_pred = self._uni_pred_cache[:, best_uni_i].astype(np.float32)

                    else:
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


                best_pair_idx = None
                best_pair_feats = None
                best_pair_loss_cached = np.inf
                if self.group_mode in ("pairwise", "mixed") and len(self._pair_meta) > 0:
                    if self._pair_pred_cache is not None:
                        valid_cols = np.arange(self._pair_pred_cache.shape[1], dtype=int)

                        if not self.allow_pair_reuse and len(self._used_pair_tuples) > 0:
                            mask = np.ones(valid_cols.size, dtype=bool)
                            for k, col in enumerate(valid_cols):
                                meta_idx = int(self._pair_cache_meta_idx[col])
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
                                    best_pair_idx = int(cols[k_rel])
                                    meta_idx = int(self._pair_cache_meta_idx[best_pair_idx])
                                    best_pair_feats = self._pair_meta[meta_idx]

                    else:
                        for k_idx, (ii, jj) in enumerate(self._pair_meta):
                            if (not self.allow_pair_reuse) and ((min(ii, jj), max(ii, jj)) in self._used_pair_tuples):
                                continue
                            pred = (
                                DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
                                .fit(X_tr[:, [ii, jj]], residual)
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
                if len(candidates) > 1 and (candidates[1][1] - candidates[0][1]) < 1e-4:
                    pass

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

                F_tr = F_tr + self.lr * pred_tr
                stage_gain = base_loss_tr - float(loss_after)
                self.stages_.append(_Stage(feats=feats, tree=tree, kind=chosen_kind, score=stage_gain))

                if self.verbose >= 2:
                    logger.debug(f"[Iter {t}/{T}] kind={chosen_kind}, features={feats}, train_gain={stage_gain:.4f}")

                if pbar is not None:
                    if (t == start_t) or (t % 25 == 0):
                        if chosen_kind == "uni":
                            pbar.set_postfix_str(f"uni f{feats[0]} gain={stage_gain:.4f}")
                        else:
                            pbar.set_postfix_str(f"pair ({feats[0]},{feats[1]}) gain={stage_gain:.4f}")
                    pbar.update(1)

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

            if self.early_stopping and best_iter < len(self.stages_):
                self.stages_ = self.stages_[:best_iter]
                self.best_iteration_ = best_iter
                self.best_score_ = best_score
                if self.verbose >= 1:
                    logger.info(f"Final model truncated at iter {best_iter} (best_score={best_score:.6f})")

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

            if pbar is not None:
                pbar.close()

            self.n_estimators_ = len(self.stages_)
            return self

        finally:
            logger.setLevel(_prev_level)

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        return self._raw_margin(X)

    def predict_proba(self, X):
        z = self.decision_function(X)
        p1 = _sigmoid(z)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: ArrayLike) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


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


import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Optional, Union, Dict
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss


def _expit(x):
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def margin_from_factor_offset(S, factor, offset):
    # z = (S - offset)/factor  -> margin/logit uncalibrated
    return (np.asarray(S, float) - float(offset)) / float(factor)

def proba_from_factor_offset(S, factor, offset):
    return _expit(margin_from_factor_offset(S, factor, offset))

def _to_inf(v, default):
    if v is None: return default
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ('-inf', '-infty', '-infinite'): return -np.inf
        if s in ('inf', '+inf', 'infty', 'infinite'): return np.inf
    return float(v)


class ScorecardCalibratorCV:
    def __init__(self, method='isotonic', n_splits=5, random_state=42,
                 class_weight=None, max_iter=2000, factor=None, offset=None,
                 refit_full=True, max_rules=None):
        assert method in ('platt', 'isotonic')
        self.method = method
        self.n_splits = n_splits
        self.random_state = random_state
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.factor = factor
        self.offset = offset
        self.refit_full = refit_full
        self.max_rules = max_rules

        self.calibrator_ = None # ('platt', w, b) or ('isotonic', iso)
        self.p1_oof_ = None
        self.metrics_ = {}

    def _to_margin(self, S):
        if self.factor is not None and self.offset is not None:
            return margin_from_factor_offset(S, self.factor, self.offset)
        return np.asarray(S, float)

    def _to_prob01(self, S):
        if self.factor is not None and self.offset is not None:
            return proba_from_factor_offset(S, self.factor, self.offset)
        S = np.asarray(S, float)
        return np.clip(S, 1e-6, 1-1e-6)

    def fit(self, S, y):
        S = np.asarray(S).reshape(-1)
        y = np.asarray(y).reshape(-1)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.random_state)

        p1_oof = np.zeros_like(S, dtype=float)

        for tr_idx, va_idx in skf.split(S, y):
            S_tr, S_va = S[tr_idx], S[va_idx]
            y_tr = y[tr_idx]

            if self.method == 'platt':
                X_tr = self._to_margin(S_tr).reshape(-1, 1)
                lr = LogisticRegression(penalty='l2', solver='saga',
                                        class_weight=self.class_weight,
                                        max_iter=self.max_iter)
                lr.fit(X_tr, y_tr)
                w = float(lr.coef_[0, 0]); b = float(lr.intercept_[0])
                p1_oof[va_idx] = _expit(w * self._to_margin(S_va) + b)

            else:
                iso = IsotonicRegression(out_of_bounds='clip',
                                         y_min=0.0, y_max=1.0)
                iso.fit(self._to_prob01(S_tr), y_tr)
                p1_oof[va_idx] = iso.predict(self._to_prob01(S_va))

        self.p1_oof_ = p1_oof
        self.metrics_ = {
            'auc_oof'   : float(roc_auc_score(y, p1_oof)),
            'brier_oof' : float(brier_score_loss(y, p1_oof)),
            'logloss_oof': float(log_loss(y, p1_oof))
        }

        if self.refit_full:
            if self.method == 'platt':
                lr = LogisticRegression(penalty='l2', solver='saga',
                                        class_weight=self.class_weight,
                                        max_iter=self.max_iter)
                lr.fit(self._to_margin(S).reshape(-1,1), y)
                w = float(lr.coef_[0, 0]); b = float(lr.intercept_[0])
                self.calibrator_ = ('platt', w, b)
            else:
                iso = IsotonicRegression(out_of_bounds='clip',
                                         y_min=0.0, y_max=1.0)
                iso.fit(self._to_prob01(S), y)
                self.calibrator_ = ('isotonic', iso)
        else:
            self.calibrator_ = None

        return self

    def predict_proba(self, S):
        S = np.asarray(S).reshape(-1)
        if self.calibrator_ is None:
            raise RuntimeError("Calibrator not found.")
        kind = self.calibrator_[0]
        if kind == 'platt':
            _, w, b = self.calibrator_
            p1 = _expit(w * self._to_margin(S) + b)
        else:
            _, iso = self.calibrator_
            p1 = iso.predict(self._to_prob01(S))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, S, threshold=0.5):
        p1 = self.predict_proba(S)[:, 1]
        return (p1 >= float(threshold)).astype(int)

@dataclass
class Scorecard:
    rules_df_points: pd.DataFrame
    factor: float
    offset: float
    base_points: int
    bounds_col: str = 'bounds'
    on_missing: str = 'error' # 'error' | 'ignore'
    return_sparse: bool = True
    calibrator_: Optional[ScorecardCalibratorCV] = None 

    @staticmethod
    def extract_stage_rules_structured(clf, feature_names, X: pd.DataFrame, y: np.ndarray, min_support: int = 10):
        from sklearn.tree import _tree
        rules = []
        n = len(X)

        for stage_idx, stage in enumerate(clf.stages_):
            tree_ = stage.tree.tree_
            feats_local2global = stage.feats

            def feat_name(local_idx):
                if local_idx == _tree.TREE_UNDEFINED:
                    return "undefined!"
                return feature_names[feats_local2global[local_idx]]

            def recurse(node, conds):
                f = tree_.feature[node]
                if f != _tree.TREE_UNDEFINED:
                    name = feat_name(f)
                    thr = float(tree_.threshold[node])
                    recurse(tree_.children_left[node],  conds + [(name, "<=", thr)])
                    recurse(tree_.children_right[node], conds + [(name, ">",  thr)])
                else:
                    bounds: Dict[str, Dict[str, float]] = {}
                    for name, op, thr in conds:
                        b = bounds.setdefault(name, {"lb": -np.inf, "ub": np.inf})
                        if op == ">":
                            b["lb"] = max(b["lb"], thr)
                        else:
                            b["ub"] = min(b["ub"], thr)

                    mask = np.ones(n, dtype=bool)
                    pretty = []
                    used_feats = []
                    for name, b in bounds.items():
                        lb, ub = b["lb"], b["ub"]
                        col = name
                        used_feats.append(name)
                        if np.isneginf(lb) and np.isposinf(ub):
                            pretty.append(f"{name}: any")
                        elif np.isneginf(lb):
                            pretty.append(f"{name} <= {ub:.12f}")
                            mask &= (X[col] <= ub)
                        elif np.isposinf(ub):
                            pretty.append(f"{name} > {lb:.12f}")
                            mask &= (X[col] >  lb)
                        else:
                            if not (lb < ub):
                                return
                            pretty.append(f"{lb:.12f} < {name} <= {ub:.12f}")
                            mask &= (X[col] > lb) & (X[col] <= ub)

                    if not bounds:
                        pretty.append("TRUE")
                        mask &= True

                    support = int(mask.sum())
                    if support == 0:
                        return

                    pos_rate = float(y[mask].mean()) if support > 0 else np.nan
                    weight_leaf = float(tree_.value[node][0][0])
                    stage_lr = getattr(stage, "learning_rate", 1.0)
                    weight_eff = weight_leaf * stage_lr
                    rules.append({
                        "stage": stage_idx,
                        "kind": stage.kind,
                        "features": tuple(sorted(used_feats)) if used_feats else tuple(),
                        "bounds": {k: dict(v) for k,v in bounds.items()},
                        "expr": " AND ".join(pretty),
                        "weight": weight_eff,
                        "support": support,
                        "pos_rate": pos_rate
                    })

            recurse(0, [])

        rules_df = pd.DataFrame(rules).sort_values(["stage","features","expr"]).reset_index(drop=True)

        # if min_support < 1, interpreted as percentage (fraction of the dataset)
        if 0 < min_support < 1:
            abs_min_support = int(np.ceil(min_support * n))
        else:
            abs_min_support = int(min_support)

        rules_df = rules_df[rules_df["support"] >= abs_min_support].reset_index(drop=True)

        return rules_df

    @staticmethod
    def weights_to_points(rules_df, *,
                        PDO=50, score0=100, odds0=50,
                        base_log_odds=None):
        """
        rules_df: (from extract_stage_rules_structured) with 'weight' and 'stage'.
        PDO, score0, odds0: parameters to define the score scale.
        base_log_odds: if F(x) already includes the base term, you can leave it None.
                    If you want to make it explicit: base_log_odds = log(p0/(1-p0)).
        Returns: rules_df with 'points', and scaling constants (factor, offset, base_points).
        """
        factor = PDO / np.log(2.0)
        offset = score0 - factor * np.log(float(odds0))

        df = rules_df.copy()
        df["lr"] = df["stage"].map(lambda s: 1.0)
        df["weight_eff"] = df["weight"] * df["lr"]
        df["points"] = np.rint(factor * df["weight_eff"]).astype(int)

        base_points = int(np.rint(offset + factor * (base_log_odds or 0.0)))
        df = df[df["points"] != 0].reset_index(drop=True)
        return df, float(factor), float(offset), int(base_points)

    @classmethod
    def from_model(cls,
                   clf: Any,
                   X_train: pd.DataFrame,
                   y_train: np.ndarray,
                   *,
                   feature_names: Optional[pd.Index] = None,
                   min_support: Union[int, float] = 0.005,
                   max_rules: Optional[int] = None,
                   PDO: int = 50,
                   score0: float = 0.0,
                   odds0: Optional[float] = None,
                   lr_by_stage: Optional[Union[float, Dict[int, float]]] = 1.0,
                   bounds_col: str = 'bounds',
                   on_missing: str = 'ignore',
                   return_sparse: bool = True
                  ) -> "Scorecard":
        if feature_names is None:
            feature_names = getattr(clf, "feature_names_in_", None)
            if feature_names is None:
                raise ValueError("feature_names non forniti e non presenti nel modello.")
        feature_names = pd.Index(feature_names)

        rules_df = cls.extract_stage_rules_structured(
            clf, feature_names, X_train, y_train, min_support=min_support
        )

        p0 = float(np.asarray(y_train).mean())
        base_log_odds = np.log(p0/(1-p0))
        if odds0 is None:
            odds0 = p0/(1-p0)

        rules_df_points, factor, offset, base_points = cls.weights_to_points(
            rules_df, PDO=PDO, score0=score0, odds0=odds0, base_log_odds=base_log_odds
        )

        sc = cls(
            rules_df_points=rules_df_points,
            factor=factor, offset=offset, base_points=base_points,
            bounds_col=bounds_col, on_missing=on_missing, return_sparse=return_sparse
        )

        if max_rules is not None:
            sc = cls.prune_rules(sc, max_rules=max_rules)

        return sc

    @staticmethod
    def make_activation_matrix(X, rules_df, bounds_col='bounds', on_missing='error', return_sparse=False, dtype=np.int8):
        import numpy as np
        n = len(X)
        m = len(rules_df)
        is_sparse = False
        if return_sparse:
            try:
                from scipy.sparse import lil_matrix
                A = lil_matrix((n, m), dtype=dtype)
                is_sparse = True
            except Exception:
                A = np.zeros((n, m), dtype=dtype)
        else:
            A = np.zeros((n, m), dtype=dtype)

        for j, (_, row) in enumerate(rules_df.iterrows()):
            spec = row[bounds_col]
            if spec is None or (isinstance(spec, float) and np.isnan(spec)):
                mask = np.zeros(n, dtype=bool)
            else:
                mask = np.ones(n, dtype=bool)
                for feat, cond in spec.items():
                    if feat not in X.columns:
                        if on_missing == 'error':
                            raise KeyError(f"Feature '{feat}' assente in X.")
                        else:
                            mask &= False
                            continue

                    col = X[feat]
                    if isinstance(cond, dict) and ('lb' in cond or 'ub' in cond):
                        lb = _to_inf(cond.get('lb', None), -np.inf)
                        ub = _to_inf(cond.get('ub', None),  np.inf)
                        lb_inc = bool(cond.get('lb_inclusive', False))
                        ub_inc = bool(cond.get('ub_inclusive', True))
                        if lb_inc:
                            m1 = (col >= lb)
                        else:
                            m1 = (col > lb)
                        if ub_inc:
                            m2 = (col <= ub)
                        else:
                            m2 = (col < ub)
                        cond_mask = m1 & m2
                    elif isinstance(cond, dict) and 'values' in cond:
                        vals = cond['values']
                        cond_mask = col.isin(vals)
                    else:
                        raise ValueError(f"Spec non riconosciuta per '{feat}': {cond}")
                    cond_mask = cond_mask.to_numpy(dtype=bool, copy=False)
                    isna_col = pd.isna(col).to_numpy()
                    cond_mask &= ~isna_col
                    mask &= cond_mask

            if is_sparse:
                idx = np.where(mask)[0]
                if len(idx):
                    A[idx, j] = 1
            else:
                A[:, j] = mask.astype(dtype, copy=False)

        if is_sparse:
            return A.tocsr()
        return A
    
    @staticmethod
    def prune_rules(sc: "Scorecard", max_rules: Optional[int]) -> "Scorecard":
        if not max_rules or max_rules is None:
            return sc
        df = sc.rules_df_points.copy()
        if len(df) <= max_rules:
            return sc
        score = df["points"].abs() * (df["support"].clip(lower=1))
        keep_idx = score.nlargest(int(max_rules)).index
        pruned = df.loc[keep_idx].sort_index().reset_index(drop=True)
        sc.rules_df_points = pruned
        return sc

    @property
    def points_(self) -> np.ndarray:
        return self.rules_df_points["points"].to_numpy()

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        score = (base_points + sum of points of active rules).
        """
        A = self.make_activation_matrix(
                X,
                self.rules_df_points,
                bounds_col=self.bounds_col,
                on_missing=self.on_missing,
                return_sparse=self.return_sparse
            )
        pts = self.points_
        S = int(self.base_points) + (A @ pts)
        return np.asarray(S).reshape(-1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        S = self.predict_scores(X)
        if self.calibrator_ is not None:
            return self.calibrator_.predict_proba(S)
        p1 = proba_from_factor_offset(S, self.factor, self.offset)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= float(threshold)).astype(int)

    def fit_calibrator(self, X: pd.DataFrame, y: np.ndarray, *,
                    method: str = 'platt', n_splits: int = 5,
                    random_state: int = 42, class_weight=None, max_iter: int = 2000):

        S = self.predict_scores(X)
        cal = ScorecardCalibratorCV(method=method, n_splits=n_splits,
                                    random_state=random_state, class_weight=class_weight,
                                    max_iter=max_iter,
                                    factor=self.factor, offset=self.offset)
        cal.fit(S, y)
        self.calibrator_ = cal
        return self


    def rules_table(self) -> pd.DataFrame:
        return self.rules_df_points.copy()

def rulecard_complexity(sc, X: pd.DataFrame) -> dict:
    rules = sc.rules_table()
    n_rules = int(len(rules))

    if n_rules > 0:
        if "features" in rules.columns:
            lists = rules["features"].apply(
                lambda xs: list(map(str, xs)) if isinstance(xs, (list, tuple, set)) else []
            ).tolist()
        else:
            bcol = getattr(sc, "bounds_col", "bounds")
            lists = rules[bcol].apply(
                lambda b: list(map(str, getattr(b, "keys", lambda: [])()))
            ).tolist()
        unique_features = sorted(set().union(*lists)) if lists else []
    else:
        unique_features = []

    n_unique_features = len(unique_features)

    if n_rules > 0:
        A = sc.make_activation_matrix(
            X, rules,
            bounds_col=getattr(sc, "bounds_col", "bounds"),
            on_missing=getattr(sc, "on_missing", "ignore"),
            return_sparse=True
        )
        if hasattr(A, "getnnz"):
            active_rules_per_sample = np.asarray(A.getnnz(axis=1)).reshape(-1)
        else:
            active_rules_per_sample = A.sum(axis=1).astype(int)
        active_rules_mean = float(np.mean(active_rules_per_sample))
    else:
        active_rules_mean = 0.0

    return dict(
        n_rules=n_rules,
        n_unique_features=n_unique_features,
        active_rules_mean=active_rules_mean
    )


from utils import get_available_datasets, load_preprocess, get_feature_importance
from tqdm import tqdm
from sklearn.model_selection import train_test_split

COMP_MAP = {
    "n_rules": "Total Rules",
    "n_unique_features": "Unique Features",
    "active_rules_mean": "Avg Active Rules",
}

best_config_dataset = pd.read_csv("05_1_best_configurations.csv")

best_config_dataset = best_config_dataset.rename(columns={"Unnamed: 0": "dataset"})
best_configs = {row["dataset"]: row["best_params"] for _, row in best_config_dataset.iterrows()}

sc_best_config_dataset = pd.read_csv("06_scorecard_best_configurations.csv")
sc_best_config_dataset = sc_best_config_dataset.rename(columns={"Unnamed: 0": "dataset"})
sc_best_configs = {}
for _, row in sc_best_config_dataset.iterrows():
    params = row[["min_support", "PDO", "max_rules", "calibrator"]].to_dict()

    if pd.isna(params["max_rules"]):
        params["max_rules"] = None
    else:
        params["max_rules"] = int(params["max_rules"])

    sc_best_configs[row["dataset"]] = params


results = {}
for name, info in get_available_datasets().items():

    data = load_preprocess(name)
    X, y = data["X"], data["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    metrics = defaultdict(list)

    if name in best_configs:
        params = ast.literal_eval(best_configs[name])
    else:
        params = {"max_depth": 3, "lr": 1.0, "group_mode": "mixed", "selection": "greedy"}
    
    if name in sc_best_configs:
        sc_params = sc_best_configs[name]
    else:
        sc_params = {"min_support": 0.005, "PDO": 50, "max_rules": None, "calibrator": "isotonic"}

    clf = Model2D(
        max_depth=params["max_depth"],
        lr=params["lr"],
        group_mode=params["group_mode"],
        selection=params["selection"],
        early_stopping=True,
        early_stopping_metric="auc",
        feature_importance_fn=get_feature_importance,
        verbose=1
    )
    start = time.time()
    clf.fit(X_train, y_train)

    sc = Scorecard.from_model(
        clf,
        X_train, y_train,
        feature_names=getattr(clf, "feature_names_in_", X_train.columns),
        min_support=sc_params["min_support"],
        PDO=sc_params["PDO"],
        max_rules=sc_params["max_rules"],
        bounds_col='bounds',
        on_missing='ignore',
        return_sparse=True
    )
    sc.fit_calibrator(X_train, y_train, method=sc_params["calibrator"], n_splits=5)
    end = time.time()

    comp = rulecard_complexity(sc, X_test)

    for src, dst in COMP_MAP.items():
        metrics[dst].append(comp.get(src, np.nan))

    S_test = sc.predict_scores(X_test)
    proba_test = sc.predict_proba(X_test)
    yhat_test = sc.predict(X_test, 0.5)

    acc = accuracy_score(y_test, yhat_test)
    prec = precision_score(y_test, yhat_test)
    rec = recall_score(y_test, yhat_test)
    f1 = f1_score(y_test, yhat_test)
    bal_acc = balanced_accuracy_score(y_test, yhat_test)
    mcc = matthews_corrcoef(y_test, yhat_test)
    error_rate = 1 - acc
    auc = roc_auc_score(y_test, proba_test[:, 1])

    metrics["Time Training (s)"].append(end - start)
    metrics["Accuracy (%)"].append(acc * 100.0)
    metrics["Precision (%)"].append(prec * 100.0)
    metrics["Recall (%)"].append(rec * 100.0)
    metrics["F1 Score (%)"].append(f1 * 100.0)
    metrics["Balanced Accuracy (%)"].append(bal_acc * 100.0)
    metrics["MCC (%)"].append(mcc * 100.0)
    metrics["Error Rate (%)"].append(error_rate * 100.0)
    metrics["AUC (%)"].append(auc * 100.0)

    p = X_test.shape[1]
    n_rules = comp.get("n_rules", 0)
    n_uf = comp.get("n_unique_features", 0)
    act_mean = comp.get("active_rules_mean", 0.0)

    UFU = (n_uf / p) if p > 0 else np.nan
    ARPF = min(1.0, act_mean / p) if p > 0 else np.nan
    RAR  = (act_mean / n_rules) if n_rules > 0 else 0.0

    metrics["Unique Features Utilization (UFU)"].append(UFU)
    metrics["Active Rules per Feature (ARPF)"].append(ARPF)
    metrics["Activation Ratio (RAR)"].append(RAR)

    results[name] = metrics

df_results = pd.DataFrame.from_dict(
    {name: {k: v[0] for k, v in metrics.items()} for name, metrics in results.items()},
    orient="index"
)
df_results

df_results.to_csv("06_1_scorecard_performance_results_for_competitors.csv")

SEEDS = [0, 17, 22, 36, 42]

all_results = {}

for name, info in tqdm(get_available_datasets().items()):
    data = load_preprocess(name)
    X, y = data["X"], data["y"]

    per_seed_metrics = defaultdict(list)

    for seed in SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )

        if name in best_configs:
            params = ast.literal_eval(best_configs[name])
        else:
            params = {"max_depth": 3, "lr": 1.0, "group_mode": "mixed", "selection": "greedy"}
        
        if name in sc_best_configs:
            sc_params = sc_best_configs[name]
        else:
            sc_params = {"min_support": 0.005, "PDO": 50, "max_rules": None, "calibrator": "isotonic"}

        clf = Model2D(
            max_depth=params["max_depth"],
            lr=params["lr"],
            group_mode=params["group_mode"],
            selection=params["selection"],
            early_stopping=True,
            early_stopping_metric="auc",
            feature_importance_fn=get_feature_importance,
            verbose=0
        )
        start = time.time()
        clf.fit(X_train, y_train)

        sc = Scorecard.from_model(
            clf,
            X_train, y_train,
            feature_names=getattr(clf, "feature_names_in_", X_train.columns),
            min_support=sc_params["min_support"],
            PDO=sc_params["PDO"],
            max_rules=sc_params["max_rules"],
            bounds_col='bounds',
            on_missing='ignore',
            return_sparse=True
        )
        sc.fit_calibrator(X_train, y_train, method=sc_params["calibrator"], n_splits=5)
        end = time.time()

        comp = rulecard_complexity(sc, X_test)
        for src, dst in COMP_MAP.items():
            per_seed_metrics[dst].append(comp.get(src, np.nan))

        S_test = sc.predict_scores(X_test)
        proba_test = sc.predict_proba(X_test)
        yhat_test = sc.predict(X_test, 0.5)

        acc = accuracy_score(y_test, yhat_test)
        prec = precision_score(y_test, yhat_test)
        rec = recall_score(y_test, yhat_test)
        f1 = f1_score(y_test, yhat_test)
        bal_acc = balanced_accuracy_score(y_test, yhat_test)
        mcc = matthews_corrcoef(y_test, yhat_test)
        error_rate = 1 - acc
        auc = roc_auc_score(y_test, proba_test[:, 1])

        per_seed_metrics["Time Training (s)"].append(end - start)
        per_seed_metrics["Accuracy (%)"].append(acc * 100.0)
        per_seed_metrics["Precision (%)"].append(prec * 100.0)
        per_seed_metrics["Recall (%)"].append(rec * 100.0)
        per_seed_metrics["F1 Score (%)"].append(f1 * 100.0)
        per_seed_metrics["Balanced Accuracy (%)"].append(bal_acc * 100.0)
        per_seed_metrics["MCC (%)"].append(mcc * 100.0)
        per_seed_metrics["Error Rate (%)"].append(error_rate * 100.0)
        per_seed_metrics["AUC (%)"].append(auc * 100.0)

        p = X_test.shape[1]
        n_rules = comp.get("n_rules", 0)
        n_uf = comp.get("n_unique_features", 0)
        act_mean = comp.get("active_rules_mean", 0.0)
        UFU = (n_uf / p) if p > 0 else np.nan
        ARPF = min(1.0, act_mean / p) if p > 0 else np.nan
        RAR  = (act_mean / n_rules) if n_rules > 0 else 0.0
        per_seed_metrics["Unique Features Utilization (UFU)"].append(UFU)
        per_seed_metrics["Active Rules per Feature (ARPF)"].append(ARPF)
        per_seed_metrics["Activation Ratio (RAR)"].append(RAR)    
    
    all_results[name] = {k: (np.mean(v), np.std(v)) for k, v in per_seed_metrics.items()}

df_mean = pd.DataFrame.from_dict(
    {name: {k: v[0] for k, v in metrics.items()} for name, metrics in all_results.items()},
    orient="index"
)
df_std = pd.DataFrame.from_dict(
    {name: {k: v[1] for k, v in metrics.items()} for name, metrics in all_results.items()},
    orient="index"
)

df_all_results = pd.concat([df_mean, df_std], axis=1, keys=["mean", "std"])
df_all_results = df_all_results.swaplevel(axis=1).sort_index(axis=1, level=0)

df_all_results.to_csv("06_1_scorecard_performance_results_for_competitors_mean_std.csv")

means = df_all_results.xs('mean', axis=1, level=1)
means.round(3).to_csv("06_1_SCORECARD_PERFORMANCE_SEEDS.csv")

from matplotlib.colors import TwoSlopeNorm

METRICS = [
    "Unique Features Utilization (UFU)",
    "Active Rules per Feature (ARPF)",
    "Activation Ratio (RAR)",
]

OUT = Path("figs_metriche_simpl/"); OUT.mkdir(exist_ok=True, parents=True)

df = means[METRICS].copy(deep=True)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(how="all")
df = df.loc[:, [m for m in METRICS if m in df.columns]]

def slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)

import matplotlib.patheffects as pe
def _srgb_to_linear(c):
    return c/12.92 if c <= 0.03928 else ((c+0.055)/1.055)**2.4

def rel_lum(rgba):
    r, g, b = rgba[:3]
    R, G, B = map(_srgb_to_linear, (r, g, b))
    return 0.2126*R + 0.7152*G + 0.0722*B  # W3C

def contrast_ratio(L1, L2):
    L1, L2 = (max(L1, L2), min(L1, L2))
    return (L1 + 0.05) / (L2 + 0.05)

def best_text_style(val, im, min_ratio=4.5):
    rgba = im.cmap(im.norm(val))
    L = rel_lum(rgba)
    cr_white = contrast_ratio(1.0, L)
    cr_black = contrast_ratio(L, 0.0)
    color = "white" if cr_white >= cr_black else "black"
    outline = [pe.withStroke(linewidth=1.6,
                             foreground=("black" if color=="white" else "white"))]
    bbox = None
    if max(cr_white, cr_black) < min_ratio:
        bbox = dict(boxstyle="round,pad=0.18", fc=("black" if color=="white" else "white"),
                    ec="none", alpha=0.25)
    return color, outline, bbox

def relative_luminance(rgba):
    r, g, b = rgba[:3]
    def lin(c): 
        return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
    R, G, B = lin(r), lin(g), lin(b)
    return 0.2126*R + 0.7152*G + 0.0722*B

def pick_text_color(val, im, threshold=None):
    if threshold is None:
        threshold = relative_luminance(im.cmap(im.norm(0.5)))
    lum = relative_luminance(im.cmap(im.norm(val)))
    return "white" if lum < threshold else "black"

if len(df.columns) == 3 and len(df) > 0:
    M = df.values
    fig, ax = plt.subplots(figsize=(8, 0.5*max(3, len(df))))

    im = ax.imshow(M, aspect="auto", vmin=0, vmax=1, cmap="BrBG_r")

    ax.set_yticks(np.arange(len(df)), labels=df.index)
    ax.set_xticks(np.arange(len(df.columns)), labels=df.columns, rotation=30, ha="right")

    thr = relative_luminance(im.cmap(im.norm(0.5)))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = float(M[i, j])
            color, outline, bbox = best_text_style(val, im, min_ratio=4.5)
            txt = ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=12, path_effects=outline, bbox=bbox)

            txt.set_path_effects([pe.withStroke(linewidth=0.8,
                                                foreground=("black" if color=="white" else "white"))])

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.savefig("06_01_heatmap_UFU_ARPF_RAR.png", dpi=200)
    plt.close(fig)