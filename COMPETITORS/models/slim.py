import numpy as np
import pandas as pd

import os

from slim_python3.slim_python.create_slim_IP_gurobi import create_slim_IP_gurobi
from slim_python3.slim_python.SLIMCoefficientConstraints import SLIMCoefficientConstraints
from slim_python3.slim_python.helpers_gurobi import slimGurobiHelpers as sgh

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, balanced_accuracy_score,
                             matthews_corrcoef, roc_auc_score,
                             confusion_matrix, classification_report)

import time

from sklearn.model_selection import StratifiedKFold

'''
Expected: 60min
'''

class SLIM():
    def __init__(self, X, Y, X_names, Y_name):
        self.X = X
        self.Y = Y
        self.X_names = X_names
        self.Y_name = Y_name
    
    def get_coef_constraints(self, print_view=False):
        coef_constraints = SLIMCoefficientConstraints(variable_names=self.X_names, ub=10, lb=-10)
        coef_constraints.set_field('ub', '(Intercept)', 100)
        coef_constraints.set_field('lb', '(Intercept)', -100)
        if print_view:
            coef_constraints.view()
        return coef_constraints

    
    def slim_input_dict(self, 
                        C_0=0.01,
                        w_pos=1.0,
                        w_neg=1.0,
                        L0_min=0,
                        L0_max=float('inf'),
                        err_min=0.0,
                        err_max=1.0,
                        pos_err_min=0.0,
                        pos_err_max=1.0,
                        neg_err_min=0.0,
                        neg_err_max=1.0,
                        coef_constraints=None):
        input_dict = {
            'X': self.X,
            'Y': self.Y,
            'X_names': self.X_names,
            'Y_name': self.Y_name,
            'C_0': C_0,
            'w_pos': w_pos,
            'w_neg': w_neg,
            'L0_min': L0_min,
            'L0_max': L0_max,
            'err_min': err_min,
            'err_max': err_max,
            'pos_err_min': pos_err_min,
            'pos_err_max': pos_err_max,
            'neg_err_min': neg_err_min,
            'neg_err_max': neg_err_max,
            'coef_constraints': coef_constraints
        }
        return input_dict

    # ref: https://github.com/ustunb/slim-python/blob/master/slim_python/create_slim_IP.py
    def create_slim_model(
            self, input_dict,
            output_flag=1,
            log_to_console=1,
            log_file="",
            threads=None,
            time_limit=600,
            seed=42,
            mip_gap_abs=0.0,
            mip_gap=1e-4,
            int_feas_tol=1e-9,
            mip_focus=1,
            heuristics=0.05,
            presolve=2,
            cuts=2):

        if threads is None:
            threads = max(1, os.cpu_count() or 1)
        
        model, slim_info = create_slim_IP_gurobi(input_dict)
        model.Params.OutputFlag = output_flag
        model.Params.LogToConsole = log_to_console
        model.Params.LogFile = log_file
        model.Params.Threads = threads
        model.Params.TimeLimit = time_limit
        model.Params.Seed = seed
        model.Params.MIPGapAbs = mip_gap_abs
        model.Params.MIPGap = mip_gap
        model.Params.IntFeasTol = int_feas_tol
        model.Params.MIPFocus = mip_focus
        model.Params.Heuristics = heuristics
        model.Params.Presolve = presolve
        model.Params.Cuts = cuts

        return model, slim_info
    
    def optimization_results(self, model, slim_info, X, Y, coef_constraints):
        model.optimize()

        try:
            sgh.check_slim_IP_output(model, slim_info, X, Y, coef_constraints)
        except Exception as e:
            print(f"Error during optimization: {e}")
            pass

        slim_results = sgh.get_slim_summary(model, slim_info, X, Y)

        metrics_dict = {
            'Accuracy (%)': 100 * slim_results['accuracy'],
            'Precision (%)': 100 * slim_results['precision'],
            'Recall (%)': 100 * slim_results['recall'],
            'F1 Score (%)': 100 * slim_results['f1'],
            'Balanced Accuracy (%)': 100 * slim_results['balanced_accuracy'],
            'MCC (%)': 100 * slim_results['mcc'],
            'Error Rate (%)': 100 * slim_results['error_rate'],
            'True Positive Rate (%)': 100 * slim_results['true_positive_rate'],
            'False Positive Rate (%)': 100 * slim_results['false_positive_rate'],
            'AUC (%)': 100 * slim_results['auc'] if slim_results['auc'] is not None else None,
            'True Positives': slim_results['true_positives'],
            'False Positives': slim_results['false_positives'],
            'True Negatives': slim_results['true_negatives'],
            'False Negatives': slim_results['false_negatives']
        }

        evaluation_df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'Value'])

        return slim_results, evaluation_df, slim_results['classification_report']

import re
import pandas as pd
import numpy as np

def parse_scorecard_string(scorecard_string):
    rules = []
    pattern_coef = re.compile(r'^\|\s*(\S.*?)\s*\|\s*([-+]?\d+)\s+points\s*\|')
    pattern_interc = re.compile(r'^\|\s*\(Intercept\)\s*\|\s*([-+]?\d+)\s+points\s*\|')
    intercept = 0
    for line in scorecard_string.splitlines():
        m0 = pattern_interc.match(line)
        if m0:
            intercept = int(m0.group(1)); continue
        m = pattern_coef.match(line)
        if m:
            feature = m.group(1).strip()
            point = int(m.group(2))
            rules.append((feature, point))
    return intercept, rules

_ADD_ROWS_RE = re.compile(r"ADD POINTS FROM ROWS\s+1\s+to\s+(\d+)", re.IGNORECASE)

def extract_model_size(res_dict):
    """
    Returns L0 = number of active rules (excluding intercept).
    Order:
      1) parse "ADD POINTS FROM ROWS 1 to N" in string_model
      2) count lines (| feature | +/-X points |) in string_model
      3) from coef vector (excluding index 0 if intercept)
    """
    sm = res_dict.get('string_model') or ""
    if sm:
        m = _ADD_ROWS_RE.search(sm)
        if m:
            return int(m.group(1))

        # fallback: count lines with points
        cnt = 0
        for line in sm.splitlines():
            # line: | some_feature |   3 points |
            if re.match(r"^\|\s*(?!\(Intercept\))\S.*\|\s*[-+]?\d+\s+points\s*\|", line):
                cnt += 1
        if cnt > 0:
            return cnt

    # altro fallback: coefs
    for key in ('coefs', 'coefficients', 'coef_'):
        if key in res_dict and res_dict[key] is not None:
            arr = np.asarray(res_dict[key]).ravel()
            if arr.size >= 1:
                return int((np.abs(arr[1:]) > 0).sum())
    raise KeyError("model_size")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_from_scorecard_rules(X, intercept, rules, return_proba=False):
    score = np.full(X.shape[0], intercept, dtype=float)
    for feature, point in rules:
        if feature not in X.columns:
            raise ValueError(f"Feature '{feature}' non trovata in X.columns")
        score += X[feature].astype(float) * point
    preds = (score >= 0).astype(int) # threshold 0 (ref original paper)
    if return_proba:
        proba = 1 / (1 + np.exp(-score))
        return preds, score, proba
    return preds, score

def run_slim(X, Y, X_names, Y_name, dataset_name, time_limit=600, verbose=1, threads=None, C0=0.01, mip_gap=0.0):
    slim = SLIM(X, Y, X_names, Y_name)
    coef_constraints = slim.get_coef_constraints()

    # only adult e haberman (ref: Ustun et al. 2019)
    Np = int((Y == 1).sum()); Nn = int((Y == -1).sum()); N = len(Y)
    w_pos = (Nn / N) if dataset_name.lower() == "adult" or dataset_name.lower() == "haberman" else 1.0
    w_neg = (Np / N) if dataset_name.lower() == "adult" or dataset_name.lower() == "haberman" else 1.0

    slim_input = slim.slim_input_dict(
        C_0=C0,
        w_pos=w_pos,
        w_neg=w_neg,
        coef_constraints=coef_constraints
    )
    slim_input['Y_name'] = Y_name

    model, slim_info = slim.create_slim_model(
        slim_input,
        time_limit=time_limit,
        heuristics=0.9,
        presolve=2,
        log_to_console=verbose,
        threads=threads,
        mip_gap=mip_gap,
        mip_gap_abs=0.0
    )

    t0 = time.time()
    results, evaluation_df, classification_report = slim.optimization_results(
        model, slim_info, slim.X, slim.Y, coef_constraints
    )
    elapsed_time = time.time() - t0
    evaluation_df.loc[len(evaluation_df.index)] = ['Time Training (s)', elapsed_time]

    try:   results['solver_obj'] = float(model.ObjVal) if model.SolCount > 0 else None
    except Exception: results['solver_obj'] = None
    try:   results['solver_bound'] = float(model.ObjBound)
    except Exception: results['solver_bound'] = None
    try:   results['solver_gap(%)'] = float(model.MIPGap)*100.0 if model.MIPGap is not None else None
    except Exception: results['solver_gap(%)'] = None
    try:   results['solver_runtime(s)'] = float(model.Runtime)
    except Exception: results['solver_runtime(s)'] = elapsed_time

    try:
        results['model_size'] = extract_model_size(results)
    except Exception:
        pass

    classification_report_df = pd.DataFrame(classification_report).T
    return results, evaluation_df, classification_report_df


def evaluate_slim(results, evaluation_df, X_test, Y_test):
    intercept, rules = parse_scorecard_string(results['string_model'])
    time_training = float(evaluation_df[evaluation_df['Metric'] == 'Time Training (s)']['Value'].values[0])

    start_test = time.time()
    y_pred, score, y_proba = predict_from_scorecard_rules(X_test, intercept, rules, return_proba=True)
    time_testing = time.time() - start_test

    y_true = ((Y_test + 1) // 2).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    err_rate = 1 - acc
    auc = roc_auc_score(y_true, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
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

    report = classification_report(y_true, y_pred, output_dict=True)
    return metrics, report

def cv_slim_path(X, Y, X_names, Y_name, dataset_name, C0_grid=None, n_splits=10,
                 time_limit=600, threads=None, verbose=0, selection="min_error"):
    """
    selection:
      - "min_error": selects C0 with minimum mean test error
      - "1se_size":  selects the smallest model among those with mean error <= (best_mean + best_sd)
    """
    if C0_grid is None:
        C0_grid = [0.01, 0.075, 0.05, 0.025, 0.001, 0.9/(X.shape[0]*X.shape[1])]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_errors = {}
    cv_sizes  = {}

    for C0 in C0_grid:
        fold_err, fold_sizes = [], []
        print(f"[CV] {dataset_name} | C0={C0} ...")
        for fi, (tr, te) in enumerate(skf.split(X, (Y==1).astype(int)), start=1):
            try:
                res, eval_df, _ = run_slim(X[tr], Y[tr], X_names, Y_name, dataset_name,
                                           time_limit=time_limit, verbose=verbose, threads=threads, C0=C0)
                # holdout
                intercept, rules = parse_scorecard_string(res['string_model'])
                Xte_df = pd.DataFrame(X[te], columns=X_names)
                y_pred, _, _ = predict_from_scorecard_rules(Xte_df, intercept, rules, return_proba=True)
                y_true = ((Y[te] + 1)//2).astype(int)
                err = 1 - accuracy_score(y_true, y_pred)
                fold_err.append(err)

                try:
                    ms = extract_model_size(res)
                except Exception:
                    ms = np.nan
                fold_sizes.append(ms)

                print(f"  - fold {fi:02d}: err={err:.4f} | size={ms} | gap%={res.get('solver_gap(%)')}")
            except Exception as e:
                print(f"  ! fold {fi:02d} FAILED: {e}")
                fold_err.append(np.nan)
                fold_sizes.append(np.nan)

        cv_errors[C0] = (float(np.nanmean(fold_err)), float(np.nanstd(fold_err)))
        if np.all(np.isnan(fold_sizes)):
            cv_sizes[C0] = (-1, -1)
        else:
            cv_sizes[C0] = (int(np.nanmin(fold_sizes)), int(np.nanmax(fold_sizes)))

        print(f"[CV] {dataset_name} | C0={C0} → mean_err={cv_errors[C0][0]:.4f}±{cv_errors[C0][1]:.4f} | size_range={cv_sizes[C0]}")

    if selection == "1se_size":
        best_c_by_err = min(cv_errors, key=lambda c: cv_errors[c][0])
        thr = cv_errors[best_c_by_err][0] + cv_errors[best_c_by_err][1]
        candidates = [c for c in C0_grid if cv_errors[c][0] <= thr]
        best_C0 = min(candidates, key=lambda c: (cv_sizes[c][1], cv_errors[c][0])) # prefer smaller size, then smaller error
    else:
        best_C0 = min(C0_grid, key=lambda c: cv_errors[c][0])

    res_final, eval_df_final, _ = run_slim(X, Y, X_names, Y_name, dataset_name,
                                           time_limit=time_limit, verbose=verbose, threads=threads, C0=best_C0)
    try:
        final_ms = extract_model_size(res_final)
    except Exception:
        final_ms = None

    return {
        'best_C0': best_C0,
        'cv_test_error_mean': cv_errors[best_C0][0],
        'cv_test_error_std':  cv_errors[best_C0][1],
        'cv_model_size_range': cv_sizes[best_C0],
        'final_model_size': final_ms,
        'final_string_model': res_final.get('string_model', ''),
    }

def load_processed(name, target_col):
    data_dir = "DATA/slim_processed/"
    data = pd.read_csv(os.path.join(data_dir, f"{name}_processed.csv"))
    X = data.drop(columns=[target_col])
    y = data[target_col]

    y = y.map({0: -1, 1: 1})

    X_values = X.values
    y_values = y.values
    feat_names = X.columns.tolist()

    return X_values, y_values, feat_names



