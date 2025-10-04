from imodels import FIGSClassifier
from sklearn.model_selection import train_test_split
import time
import multiprocessing as mp
import os, joblib, tempfile

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, balanced_accuracy_score,
                             matthews_corrcoef, roc_auc_score,
                             confusion_matrix, classification_report)


def fit_figs_in_process(X_train, Y_train, X_names, max_rules, max_trees, path_q):
    model = FIGSClassifier(max_rules=max_rules, max_trees=max_trees, random_state=42)
    model.fit(X_train, Y_train, feature_names=list(X_names))
    model_path = os.path.join(tempfile.gettempdir(), "figs_model.joblib")
    joblib.dump(model, model_path, compress=3)
    path_q.put(model_path)

def run_figs(X, Y, X_names, Y_name, dataset_name, max_rules=None, max_trees=None, time_limit=120):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    path_q = mp.Queue()
    p = mp.Process(target=fit_figs_in_process, args=(X_train, Y_train, X_names, max_rules, max_trees, path_q))

    start_train = time.time()
    p.start(); p.join(timeout=time_limit)

    if p.is_alive():
        p.terminate()
        p.join()
        print(f"[TIMEOUT] Training stopped after {time_limit} seconds.")
        return {
            'model': None,
            'X': X,
            'Y': Y,
            'X_train': X_train,
            'X_test': X_test,
            'Y_train': Y_train,
            'Y_test': Y_test,
            'X_names': X_names,
            'Y_name': Y_name,
            'dataset_name': dataset_name,
            'time_training': time_limit,
            'feature_importance': None,
            'tree_text': "[TIMEOUT]"
        }

    time_training = time.time() - start_train
    model_path = path_q.get_nowait()
    model = joblib.load(model_path)

    try:
        tree_text = model.print_tree(X_train, Y_train, feature_names=list(X_names))
    except Exception:
        tree_text = "[unavailable]"

    return {
        'model': model, 'X': X, 'Y': Y,
        'X_train': X_train, 'X_test': X_test,
        'Y_train': Y_train, 'Y_test': Y_test,
        'X_names': X_names, 'Y_name': Y_name,
        'dataset_name': dataset_name,
        'time_training': time_training,
        'feature_importance': getattr(model, 'feature_importances_', None),
        'tree_text': tree_text
    }


def evaluate_figs(model, X_test, Y_test, time_training):
    if model is None:
        print("[WARN] Model is None (check timeout).")
        return {
            'Accuracy (%)': None,
            'Precision (%)': None,
            'Recall (%)': None,
            'F1 Score (%)': None,
            'Balanced Accuracy (%)': None,
            'MCC (%)': None,
            'Error Rate (%)': None,
            'AUC (%)': None,
            'True Positive Rate (%)': None,
            'False Positive Rate (%)': None,
            'True Positives': None,
            'False Positives': None,
            'True Negatives': None,
            'False Negatives': None,
            'Time Training (s)': time_training,
            'Time Testing (s)': None
        }, f"[TIMEOUT (train > {time_training}s)]"
    
    try:
        if hasattr(model, 'feature_importances_'):
            used_features_idx = (model.feature_importances_ > 0)
            n_features_used = used_features_idx.sum()
        elif hasattr(model, 'features_'):
            n_features_used = len(set(model.features_))
        else:
            n_features_used = None
        
        n_features_originali = X_test.shape[1]
        sparsity_ratio = None
        if n_features_used is not None and n_features_originali > 0:
            sparsity_ratio = 1 - (n_features_used / n_features_originali)
    except Exception as e:
        print(f"[WARN] Impossibile calcolare Sparsity Ratio: {e}")
        sparsity_ratio = None

    start_test = time.time()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    time_testing = time.time() - start_test

    acc = accuracy_score(Y_test, y_pred)
    prec = precision_score(Y_test, y_pred)
    rec = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    bal_acc = balanced_accuracy_score(Y_test, y_pred)
    mcc = matthews_corrcoef(Y_test, y_pred)
    err_rate = 1 - acc
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
        'Time Testing (s)': time_testing,
        'Sparsity Ratio': sparsity_ratio
    }

    report = classification_report(Y_test, y_pred, output_dict=True)
    
    return metrics, report