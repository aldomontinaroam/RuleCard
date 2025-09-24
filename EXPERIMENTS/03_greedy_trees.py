from utils import load_preprocess, get_available_datasets, get_feature_importance
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.tree import DecisionTreeRegressor

from sklearn.base import BaseEstimator
import numpy as np

from scipy.stats import wilcoxon
import time
import os

from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)


class Model(BaseEstimator):
    def __init__(self, max_depth=2, lr=1.0):
        self.max_depth = max_depth
        self.lr = lr
        self.models = []
        self.feature_order = []
        self.feature_order_index = None
        self.p0 = None
        self.log_odds_p0 = None

    def fit(self, X, y):
        
        self.feature_order, _ = get_feature_importance(X, y)
        self.feature_order_index = [X.columns.get_loc(feat) for feat in self.feature_order]

        X = np.asarray(X)
        y = np.asarray(y)

        self.p0 = np.mean(y)
        self.log_odds_p0 = np.log(self.p0 / (1 - self.p0))

        F = np.full(len(y), self.log_odds_p0)
        res = y - expit(F)

        self.models = []

        for i in self.feature_order_index:
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X[:, [i]], res)

            pred = tree.predict(X[:, [i]])
            F += self.lr * pred
            res = y - expit(F)

            self.models.append((i, tree)) 

        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        F = np.full(len(X), self.log_odds_p0)

        for i, tree in self.models:
            pred = tree.predict(X[:, [i]])
            F += self.lr * pred

        return expit(F)

    def predict(self, X, threshold=0.5):
        X = np.asarray(X)
        return (self.predict_proba(X) >= threshold).astype(int)


class Model_v2(BaseEstimator):
    def __init__(self, max_depth=2, lr=1.0, n_stages=None, random_state=None):
        self.max_depth = max_depth
        self.lr = lr
        self.n_stages = n_stages
        self.random_state = random_state
        self.models = []
        self.p0 = None
        self.log_odds_p0 = None
        self.used_features = set()

    def _base_init(self, y):
        self.p0 = float(np.mean(y))
        eps = 1e-12
        self.p0 = min(max(self.p0, eps), 1 - eps)
        self.log_odds_p0 = float(np.log(self.p0 / (1 - self.p0)))

    @staticmethod
    def _logloss_from_F(y, F):
        # calcola logloss(y, sigmoid(F)) in modo stabile
        p = expit(F)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(float)
        if not (np.array_equal(np.unique(y), [0]) or
                np.array_equal(np.unique(y), [1]) or
                np.array_equal(np.unique(y), [0, 1])):
            raise ValueError("Questo classificatore supporta y binaria {0,1}.")

        n_samples, n_features = X.shape
        self.models = []
        self.used_features = set()

        # base score
        self._base_init(y)
        F = np.full(n_samples, self.log_odds_p0, dtype=float)

        iterations = n_features if self.n_stages is None else int(min(self.n_stages, n_features))

        for _ in range(iterations):
            res = y - expit(F)

            best_feat = None
            best_tree = None
            best_score = np.inf

            for i in range(n_features):
                if i in self.used_features:
                    continue
                tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
                tree.fit(X[:, [i]], res)
                pred = tree.predict(X[:, [i]])
                new_F = F + self.lr * pred
                score = self._logloss_from_F(y, new_F)
                if score < best_score:
                    best_score = score
                    best_feat = i
                    best_tree = tree

            if best_feat is None:
                break

            F += self.lr * best_tree.predict(X[:, [best_feat]])
            self.models.append((best_feat, best_tree))
            self.used_features.add(best_feat)

        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        F = np.full(len(X), self.log_odds_p0, dtype=float)
        for i, tree in self.models:
            F += self.lr * tree.predict(X[:, [i]])
        return expit(F)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def _specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    denom = tn + fp
    return np.nan if denom == 0 else tn / denom

def _safe_auc(y_true, proba):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, proba)

def _safe_pr_auc(y_true, proba):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, proba)

def _holm_correction(pvals, alpha=0.05):
    """Holm step-down; ritorna (rejected_list, adjusted_pvals_list) nell'ordine originale."""
    m = len(pvals)
    order = np.argsort(pvals)
    p_sorted = np.array(pvals)[order]
    adj_sorted = np.empty(m, dtype=float)
    for i in range(m):
        adj_sorted[i] = (m - i) * p_sorted[i]
    for i in range(1, m):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i-1])
    adj = np.empty(m, dtype=float)
    adj[order] = np.minimum(adj_sorted, 1.0)
    rejected = adj < alpha
    return rejected.tolist(), adj.tolist()

# mapping metriche
METRICS = [
    "logloss", "roc_auc", "pr_auc", "brier", "accuracy", "balanced_accuracy",
    "f1", "precision", "recall", "specificity", "mcc", "time"
]
BETTER_HIGHER = {
    "logloss": False, "roc_auc": True, "pr_auc": True, "brier": False,
    "accuracy": True, "balanced_accuracy": True, "f1": True, "precision": True,
    "recall": True, "specificity": True, "mcc": True, "time": False
}

# ========= Experiments =========
seeds = [0, 1, 2, 3, 4, 5]
custom_datasets = get_available_datasets()

per_seed_metrics_model    = {s: {} for s in seeds}
per_seed_metrics_model_v2 = {s: {} for s in seeds}

for seed in seeds:
    for name, info in custom_datasets.items():
        data = load_preprocess(name)
        X, y = data['X'], data['y']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )
        X_test_np = np.asarray(X_test)
        y_test_np = np.asarray(y_test)

        # ---------- Model (fixed) ----------
        model = Model(max_depth=2, lr=1.0)
        t0 = time.time()
        model.fit(X_train, y_train)
        t1 = time.time()
        proba_m1 = model.predict_proba(X_test_np)
        pred_m1  = (proba_m1 >= 0.5).astype(int)

        m1 = {
            "time":              float(t1 - t0),
            "logloss":           float(log_loss(y_test_np, proba_m1)),
            "roc_auc":           float(_safe_auc(y_test_np, proba_m1)),
            "pr_auc":            float(_safe_pr_auc(y_test_np, proba_m1)),
            "brier":             float(brier_score_loss(y_test_np, proba_m1)),
            "accuracy":          float(accuracy_score(y_test_np, pred_m1)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test_np, pred_m1)),
            "precision":         float(precision_score(y_test_np, pred_m1, zero_division=0)),
            "recall":            float(recall_score(y_test_np, pred_m1, zero_division=0)),
            "f1":                float(f1_score(y_test_np, pred_m1, zero_division=0)),
            "specificity":       float(_specificity(y_test_np, pred_m1)),
            "mcc":               float(matthews_corrcoef(y_test_np, pred_m1)),
        }
        per_seed_metrics_model[seed][name] = m1

        # ---------- Model_v2 (greedy) ----------
        model2 = Model_v2(max_depth=2, lr=1.0)
        t0 = time.time()
        model2.fit(X_train, y_train)
        t1 = time.time()
        proba_m2 = model2.predict_proba(X_test_np)
        pred_m2  = (proba_m2 >= 0.5).astype(int)

        m2 = {
            "time":              float(t1 - t0),
            "logloss":           float(log_loss(y_test_np, proba_m2)),
            "roc_auc":           float(_safe_auc(y_test_np, proba_m2)),
            "pr_auc":            float(_safe_pr_auc(y_test_np, proba_m2)),
            "brier":             float(brier_score_loss(y_test_np, proba_m2)),
            "accuracy":          float(accuracy_score(y_test_np, pred_m2)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test_np, pred_m2)),
            "precision":         float(precision_score(y_test_np, pred_m2, zero_division=0)),
            "recall":            float(recall_score(y_test_np, pred_m2, zero_division=0)),
            "f1":                float(f1_score(y_test_np, pred_m2, zero_division=0)),
            "specificity":       float(_specificity(y_test_np, pred_m2)),
            "mcc":               float(matthews_corrcoef(y_test_np, pred_m2)),
        }
        per_seed_metrics_model_v2[seed][name] = m2

metrics_model, metrics_model_v2 = {}, {}

for name in custom_datasets:
    vals1 = {m: [] for m in METRICS}
    vals2 = {m: [] for m in METRICS}
    for seed in seeds:
        if name in per_seed_metrics_model[seed]:
            for m in METRICS:
                vals1[m].append(per_seed_metrics_model[seed][name][m])
        if name in per_seed_metrics_model_v2[seed]:
            for m in METRICS:
                vals2[m].append(per_seed_metrics_model_v2[seed][name][m])

    if len(vals1["logloss"]) > 0 and len(vals2["logloss"]) > 0:
        metrics_model[name]    = {m: float(np.nanmean(vals1[m])) for m in METRICS}
        metrics_model_v2[name] = {m: float(np.nanmean(vals2[m])) for m in METRICS}

rows = []
for name in sorted(metrics_model.keys()):
    if name not in metrics_model_v2:
        continue
    row = {"dataset": name}
    for m in METRICS:
        v1 = metrics_model[name][m]
        v2 = metrics_model_v2[name][m]
        row[f"model_{m}"]    = v1
        row[f"model_v2_{m}"] = v2
        row[f"diff_{m}"]     = v2 - v1
    rows.append(row)

per_dataset_df = pd.DataFrame(rows).set_index("dataset")

paired = {m: {"v1": [], "v2": [], "datasets": []} for m in METRICS}
for name in metrics_model:
    if name not in metrics_model_v2:
        continue
    for m in METRICS:
        a = metrics_model[name][m]
        b = metrics_model_v2[name][m]
        if np.isfinite(a) and np.isfinite(b):
            paired[m]["v1"].append(a)
            paired[m]["v2"].append(b)
            paired[m]["datasets"].append(name)

raw_results = []
for m in METRICS:
    v1 = np.asarray(paired[m]["v1"], dtype=float)
    v2 = np.asarray(paired[m]["v2"], dtype=float)
    if len(v1) >= 1 and len(v1) == len(v2):
        alt = 'greater' if BETTER_HIGHER[m] else 'less'
        stat, p = wilcoxon(v1, v2, alternative=alt)
        med = float(np.nanmedian(v2 - v1))
        raw_results.append({
            "metric": m,
            "direction": "higher_is_better" if BETTER_HIGHER[m] else "lower_is_better",
            "alternative": alt,
            "N_datasets": int(len(v1)),
            "W_stat": float(stat),
            "p_value": float(p),
            "median_diff": med
        })
    else:
        raw_results.append({
            "metric": m,
            "direction": "higher_is_better" if BETTER_HIGHER[m] else "lower_is_better",
            "alternative": 'NA',
            "N_datasets": int(len(v1)),
            "W_stat": np.nan,
            "p_value": np.nan,
            "median_diff": np.nan
        })

wilcoxon_df = pd.DataFrame(raw_results)

wilcoxon_df["p_value_holm"] = np.nan  # float
wilcoxon_df["significant_holm"] = pd.Series([False]*len(wilcoxon_df), dtype="boolean")

mask_valid = wilcoxon_df["p_value"].notna().values
pvals = wilcoxon_df.loc[mask_valid, "p_value"].to_numpy()

if pvals.size > 0:
    rejected, p_adj = _holm_correction(pvals.tolist(), alpha=0.05)
    wilcoxon_df.loc[mask_valid, "p_value_holm"] = p_adj
    wilcoxon_df.loc[mask_valid, "significant_holm"] = pd.Series(rejected, dtype="boolean").values

wilcoxon_df = wilcoxon_df.sort_values(
    by=["significant_holm", "p_value_holm", "p_value"],
    ascending=[False, True, True],
    na_position="last"
).reset_index(drop=True)

os.makedirs("tables", exist_ok=True)
per_dataset_path = os.path.join("tables", "03_per_dataset_metrics.csv")
wilcoxon_path    = os.path.join("tables", "03_wilcoxon_summary.csv")
per_dataset_df.to_csv(per_dataset_path, float_format="%.8g")
wilcoxon_df.to_csv(wilcoxon_path, index=False, float_format="%.8g")
