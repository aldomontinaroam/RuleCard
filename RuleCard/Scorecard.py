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
