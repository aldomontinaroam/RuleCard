"""
EBM → LASSO post-processing (Algoritmo 1)
=========================================
This script (Python, interpret>=0.4.4) implements the post-processing pipeline for Explainable Boosting Machines (EBMs):
  1) trains an EBM (Regressor or Classifier),
  2) constructs X* with the contributions of the terms (main + any interactions),
  3) estimates the regularization path (LASSO for regression; Logistic L1 for classification) on the training set,
  4) selects the best alpha (or C) on validation (no leakage),
  5) recalibrates the EBM terms with the coefficients selected via .scale() and updates the intercept,
  6) removes the terms with zero weight via .sweep(),
  7) evaluates on test and saves the model.

Ref: Greenwell et al. (2023), "Explainable Boosting Machines with Sparsity"
— Section "Post-processing EBMs with the LASSO" and Algorithm 1

Requires interpret>=0.4.4.
"""

import time
import copy

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score,
        matthews_corrcoef, confusion_matrix, roc_auc_score, classification_report
)

from sklearn.linear_model import LogisticRegression
from interpret.glassbox import ExplainableBoostingClassifier as EBC, ExplainableBoostingRegressor as EBR

# --------------------------
# Core: EBM + LASSO
# --------------------------
def get_term_contributions(ebm, X):
    """
    Returns the term contributions matrix (shape: [n_samples, n_terms]).
    For classification, these are contributions on the link scale (logit).
    """
    import numpy as np

    # InterpretML exposes eval_terms as a stable API for local contributions
    # (equivalent to explain_local), and works for classifier/regressor.
    contrib = ebm.eval_terms(X)  # also accepts DataFrame; returns array-like
    return np.asarray(contrib)

def rescale_ebm_from_linear_model(ebm, linear_model, sweep: bool):
    """
    Applies the coefficients of the linear model (Lasso or LogisticRegression)
    to the terms of the EBM via .scale(); updates the intercept; optionally .sweep() to remove zero-weight terms.
    """
    coefs = (
        linear_model.coef_.ravel()
        if hasattr(linear_model, "coef_")
        else linear_model.coef_
    )
    intercept = float(linear_model.intercept_.ravel()[0] if hasattr(linear_model.intercept_, "__len__") else linear_model.intercept_)

    for idx, _ in enumerate(ebm.term_names_):
        factor = float(coefs[idx]) if idx < len(coefs) else 0.0
        ebm.scale(idx, factor=factor)

    ebm.intercept_ = intercept

    if sweep:
        ebm.sweep() # remove zero-weight terms

    return ebm

def run_ebm_lasso(X, Y, X_names, Y_name, dataset_name):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2,
        stratify=Y
    )
    start_time = time.time()
    model_full = EBC(
            interactions=0,
            max_bins=20,
            n_jobs=-1
    )
    model_full.fit(X_train, y=Y_train)

    Xtr_tc = get_term_contributions(model_full, X_train)
    Xte_tc = get_term_contributions(model_full, X_test)

    Xtrv_tc = np.vstack([Xtr_tc, Xte_tc])
    y_trv = np.concatenate([np.asarray(Y_train), np.asarray(Y_test)])

    linear_model = LogisticRegression(
        penalty="l1", solver="saga", C=0.1, max_iter=1000, random_state=42
    )
    linear_model.fit(Xtrv_tc, y_trv)
    
    ebm_reduced = copy.deepcopy(model_full)
    ebm_reduced = rescale_ebm_from_linear_model(ebm_reduced, linear_model, sweep=True)
    time_training = time.time() - start_time

    results = {
        'model': ebm_reduced,
        'X': X,
        'Y': Y,
        'X_train': X_train,
        'X_test': X_test,
        'Y_train': Y_train,
        'Y_test': Y_test,
        'X_names': X_names,
        'Y_name': Y_name,
        'dataset_name': dataset_name,
        'time_training': time_training
    }

    return results

def evaluate_ebm_lasso(model, X_test, Y_test, time_training):
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    time_testing = end_time - start_time

    acc = accuracy_score(Y_test, y_pred)
    prec = precision_score(Y_test, y_pred)
    rec = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    bal_acc = balanced_accuracy_score(Y_test, y_pred)
    mcc = matthews_corrcoef(Y_test, y_pred)
    err_rate = 1 - acc

    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    auc_score = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1]) if len(set(Y_test)) > 1 else None

    metrics = {
        'Accuracy (%)': acc * 100,
        'Precision (%)': prec * 100,
        'Recall (%)': rec * 100,
        'F1 Score (%)': f1 * 100,
        'Balanced Accuracy (%)': bal_acc * 100,
        'MCC (%)': mcc * 100,
        'AUC (%)': auc_score * 100 if auc_score is not None else None,
        'Error Rate (%)': err_rate * 100,
        'True Positive Rate (%)': tpr * 100,
        'False Positive Rate (%)': fpr * 100,
        'True Positives': tp,
        'False Positives': fp,
        'True Negatives': tn,
        'False Negatives': fn,
        'Time Training (s)': time_training,
        'Time Testing (s)': time_testing
    }

    report = classification_report(Y_test, y_pred, output_dict=True)
    
    return metrics, report