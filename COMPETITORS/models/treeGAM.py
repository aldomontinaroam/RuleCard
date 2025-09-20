from imodels import TreeGAMClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, balanced_accuracy_score,
                             matthews_corrcoef, roc_auc_score,
                             confusion_matrix, classification_report)


from sklearn.model_selection import train_test_split
import time

import pandas as pd

'''
https://csinva.io/imodels/algebraic/tree_gam.html
'''

def run_treeGAM(
        X, Y, X_names, Y_name, 
        dataset_name, 
        n_boosting_rounds=100,
        random_state=42,
        n_boosting_rounds_marginal=0,
        max_leaf_nodes_marginal=2
        ):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=random_state
    )

    m = TreeGAMClassifier(
        n_boosting_rounds=n_boosting_rounds,
        random_state=random_state,
        n_boosting_rounds_marginal=n_boosting_rounds_marginal,
        max_leaf_nodes_marginal=max_leaf_nodes_marginal,
    )
    start_time = time.time()
    m.fit(X_train, Y_train)
    time_training = time.time() - start_time

    results = {
        'model': m,
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

def evaluate_treeGAM(model, X_test, Y_test, time_training):
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
    report_df = pd.DataFrame(report).transpose()

    return metrics, report_df