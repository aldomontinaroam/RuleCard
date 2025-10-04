from sklearn.model_selection import train_test_split
import pandas as pd

from pygam import LogisticGAM, s, f
from functools import reduce
import operator

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score,
    matthews_corrcoef, confusion_matrix, classification_report, roc_auc_score
)
import time

import numpy as np
from itertools import combinations
from tqdm import tqdm
import os, time

from sklearn.feature_selection import SelectKBest, mutual_info_classif


def evaluate_gam(gam, X_test, Y_test, time_training, dataset_name):
    start_time = time.time()
    y_pred = gam.predict(X_test)
    end_time = time.time()
    time_testing = end_time - start_time

    acc = accuracy_score(Y_test, y_pred)
    prec = precision_score(Y_test, y_pred)
    rec = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    bal_acc = balanced_accuracy_score(Y_test, y_pred)
    mcc = matthews_corrcoef(Y_test, y_pred)
    err_rate = 1 - acc
    y_proba = gam.predict_proba(X_test)
    auc = roc_auc_score(Y_test, y_proba)

    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    metrics = {
        'Accuracy (%)': acc * 100,
        'Precision (%)': prec * 100,
        'Recall (%)': rec * 100,
        'F1 Score (%)': f1 * 100,
        'Balanced Accuracy (%)': bal_acc * 100,
        'MCC (%)': mcc * 100,
        'Error Rate (%)': err_rate * 100,
        'AUC (%)': auc * 100,
        'True Positive Rate (%)': tpr * 100,
        'False Positive Rate (%)': fpr * 100,
        'True Positives': tp,
        'False Positives': fp,
        'True Negatives': tn,
        'False Negatives': fn,
        'Time Training (s)': time_training,
        'Time Testing (s)': time_testing
    }

    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    metrics_df.to_csv(f"RESULTS/gam/gam_{dataset_name}.csv", index=False)

    report = classification_report(Y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"RESULTS/gam/gam_{dataset_name}_cr.csv", index=True)    
    
    return metrics


def _ensure_results_dir(path="RESULTS/gam"):
    os.makedirs(path, exist_ok=True)
    return path

def _terms_for_subset(n_cols: int):
    """Terms s(0)+s(1)+... relative to the sliced matrix (local indices)."""
    terms = [s(j) for j in range(n_cols)]
    return terms[0] if n_cols == 1 else reduce(operator.add, terms)

def _safe_auc(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan

def _slice_cols(M, feats):
    """Returns always an ndarray (n_samples, len(feats)) regardless of the type of M."""
    cols = list(feats)
    if isinstance(M, pd.DataFrame):
        return M.iloc[:, cols].to_numpy()
    return M[:, cols]

def _nC2(n: int) -> int:
    return n * (n - 1) // 2

def run_gam(
    X, Y, X_names, Y_name, dataset_name,
    random_state=42,
    top_k=10,
    selector="gam_auc", # "gam_auc" | "mutual_info"
    refit_on_trainval=True,
    verbose=True,
):
    Y = np.asarray(Y).ravel()

    if isinstance(X, pd.DataFrame):
        X = X.astype(float)
    else:
        X = np.asarray(X, dtype=float)

    X_trval, X_test, Y_trval, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=random_state, stratify=Y
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trval, Y_trval, test_size=0.25, random_state=random_state, stratify=Y_trval
    )

    n_features = X.shape[1]
    top_k = min(top_k, n_features)

    # -----------------------------
    # 1) UNIVARIATE: GAM 1D
    # -----------------------------
    results = []
    best_auc = -np.inf
    best = dict(idx=None, gam=None, time_training=None)

    total = n_features + _nC2(top_k)  # barra su uni + (poi) coppie top-K
    pbar = tqdm(total=total, desc="Training GAM candidates", disable=not verbose)

    uni_scores = np.full(n_features, np.nan)  # val AUC per ciascuna feature

    for i in range(n_features):
        feats = (i,)
        Xtr = _slice_cols(X_train, feats)
        Xvl = _slice_cols(X_val, feats)
        terms = _terms_for_subset(Xtr.shape[1])

        start = time.time()
        gam = LogisticGAM(terms).fit(Xtr, Y_train)
        tr_time = time.time() - start

        yvl_proba = gam.predict_proba(Xvl)
        auc = _safe_auc(Y_val, yvl_proba)
        uni_scores[i] = auc

        yvl_pred = (yvl_proba >= 0.5).astype(int)
        f1 = f1_score(Y_val, yvl_pred) if len(np.unique(Y_val)) > 1 else np.nan
        bal_acc = balanced_accuracy_score(Y_val, yvl_pred) if len(np.unique(Y_val)) > 1 else np.nan
        acc = accuracy_score(Y_val, yvl_pred) if len(np.unique(Y_val)) > 1 else np.nan

        results.append({
            'features_idx': feats,
            'features_names': [X_names[i]],
            'n_features': 1,
            'val_auc': auc,
            'val_f1': f1,
            'val_bal_acc': bal_acc,
            'val_acc': acc,
            'time_training_s': tr_time,
        })

        if np.isfinite(auc) and auc >= best_auc:
            best_auc = auc
            best.update(idx=feats, gam=gam, time_training=tr_time)

        pbar.update(1)
        if verbose:
            pbar.set_postfix(best_auc=f"{best_auc:.4f}" if np.isfinite(best_auc) else "nan")

    # ------------------------------------------------------
    # 2) TOP-K: selection of best univariate features
    #    option A (default): ranking by val AUC of GAM
    #    option B: SelectKBest with mutual_info_classif
    # ------------------------------------------------------
    if selector == "mutual_info":
        skb = SelectKBest(score_func=mutual_info_classif, k=top_k)
        skb.fit(X_train, Y_train)
        top_idx = np.argsort(-skb.scores_)[:top_k]
    else:
        order = np.argsort(-(np.nan_to_num(uni_scores, nan=-np.inf)))
        top_idx = order[:top_k]

    top_idx = [int(i) for i in top_idx if i is not None]
    do_pairs = len(top_idx) >= 2

    # -----------------------------------------
    # 3) PAIRWISE: only between the selected top-K
    # -----------------------------------------
    if do_pairs:
        for i, j in combinations(top_idx, 2):
            feats = (i, j)
            Xtr = _slice_cols(X_train, feats)
            Xvl = _slice_cols(X_val, feats)
            terms = _terms_for_subset(Xtr.shape[1])

            start = time.time()
            gam = LogisticGAM(terms).fit(Xtr, Y_train)
            tr_time = time.time() - start

            yvl_proba = gam.predict_proba(Xvl)
            auc = _safe_auc(Y_val, yvl_proba)

            yvl_pred = (yvl_proba >= 0.5).astype(int)
            f1 = f1_score(Y_val, yvl_pred) if len(np.unique(Y_val)) > 1 else np.nan
            bal_acc = balanced_accuracy_score(Y_val, yvl_pred) if len(np.unique(Y_val)) > 1 else np.nan
            acc = accuracy_score(Y_val, yvl_pred) if len(np.unique(Y_val)) > 1 else np.nan

            results.append({
                'features_idx': feats,
                'features_names': [X_names[i], X_names[j]],
                'n_features': 2,
                'val_auc': auc,
                'val_f1': f1,
                'val_bal_acc': bal_acc,
                'val_acc': acc,
                'time_training_s': tr_time,
            })

            if np.isfinite(auc) and auc >= best_auc:
                best_auc = auc
                best.update(idx=feats, gam=gam, time_training=tr_time)

            pbar.update(1)
            if verbose:
                pbar.set_postfix(best_auc=f"{best_auc:.4f}" if np.isfinite(best_auc) else "nan")
    else:
        pbar.update(_nC2(top_k))

    pbar.close()

    results_df = pd.DataFrame(results).sort_values(
        by=['val_auc', 'val_f1', 'n_features'], ascending=[False, False, True]
    ).reset_index(drop=True)

    outdir = _ensure_results_dir()
    results_csv = os.path.join(outdir, f"gam_candidates_{dataset_name}.csv")
    results_df.to_csv(results_csv, index=False)

    if best['idx'] is None:
        raise RuntimeError("Nessun modello valido selezionato (AUC non definita).")

    selected_idx = best['idx']
    selected_names = [X_names[i] for i in selected_idx]

    if refit_on_trainval:
        X_trval_sel = _slice_cols(X_trval, selected_idx)
        terms = _terms_for_subset(X_trval_sel.shape[1])
        start = time.time()
        best_gam = LogisticGAM(terms).fit(X_trval_sel, Y_trval)
        time_training_final = time.time() - start
    else:
        best_gam = best['gam']
        time_training_final = best['time_training']

    X_test_selected  = _slice_cols(X_test,  selected_idx)
    X_train_selected = _slice_cols(X_train, selected_idx)
    X_val_selected   = _slice_cols(X_val,   selected_idx)

    topk_table = (
        pd.DataFrame({
            'feature_idx': np.arange(n_features),
            'feature_name': X_names,
            'uni_val_auc': uni_scores
        })
        .sort_values('uni_val_auc', ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    topk_csv = os.path.join(outdir, f"gam_top{top_k}_univariate_{dataset_name}.csv")
    topk_table.to_csv(topk_csv, index=False)

    return {
        'model': best_gam,
        'selected_features_idx': selected_idx,
        'selected_features_names': selected_names,

        'X': X, 'Y': Y,
        'X_train': X_train, 'Y_train': Y_train,
        'X_val': X_val, 'Y_val': Y_val,
        'X_test': X_test, 'Y_test': Y_test,

        'X_train_selected': X_train_selected,
        'X_val_selected': X_val_selected,
        'X_test_selected': X_test_selected,

        'X_names': X_names,
        'Y_name': Y_name,
        'dataset_name': dataset_name,

        'candidate_results': results_df,
        'time_training': time_training_final,
        'candidates_csv': results_csv,
        'topk_csv': topk_csv,
        'topk_indices': top_idx,
    }
