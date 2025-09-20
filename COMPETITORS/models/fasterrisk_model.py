import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, confusion_matrix,
    classification_report, roc_auc_score
)
import time

from sklearn.model_selection import train_test_split

from fasterrisk.fasterrisk import RiskScoreOptimizer, RiskScoreClassifier

# note: create a separate venv to use this script (pip dependencies not compatible with other models scripts)

def run_fasterrisk(X, Y, X_names, Y_name, dataset_name, sparsity):
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train_np = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train
    y_train_np = Y_train.to_numpy() if hasattr(Y_train, "to_numpy") else Y_train

    m = RiskScoreOptimizer(X=X_train_np, y=y_train_np, k=sparsity)
    
    start_train = time.time()
    m.optimize()
    time_training = time.time() - start_train

    # get all top m solutions from the final diverse pool
    #arr_multiplier, arr_intercept, arr_coefficients = m.get_models() # get m solutions from the diverse pool; Specifically, arr_multiplier.shape=(m, ), arr_intercept.shape=(m, ), arr_coefficients.shape=(m, p)

    # get the first solution from the final diverse pool by passing an optional model_index; models are ranked in order of increasing logistic loss
    multiplier, intercept, coefficients = m.get_models(model_index = 0) # get the first solution (smallest logistic loss) from the diverse pool; Specifically, multiplier.shape=(1, ), intercept.shape=(1, ), coefficients.shape=(p, )

    featureNames = list(X_names) # get the feature names from the training data

    # create a classifier
    clf = RiskScoreClassifier(multiplier = multiplier, intercept = intercept, coefficients = coefficients, featureNames = featureNames)

    # get the predicted label
    y_pred = clf.predict(X = X_train)

    # get the probability of predicting y[i] with label +1
    y_pred_prob = clf.predict_prob(X = X_train)

    # compute the logistic loss
    logisticLoss_train = clf.compute_logisticLoss(X = X_train, y = Y_train)

    # get accuracy and area under the ROC curve (AUC)
    acc_train, auc_train = clf.get_acc_and_auc(X = X_train, y = Y_train)

    return {
        'optimizer': m,
        'classifier': clf,
        'X': X,
        'Y': Y,
        'X_train': X_train,
        'X_train_np': X_train_np,
        'Y_train': Y_train,
        'y_train_np': y_train_np,
        'X_test': X_test,
        'Y_test': Y_test,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob,
        'logisticLoss_train': logisticLoss_train,
        'acc_train': acc_train,
        'auc_train': auc_train,
        'featureNames': featureNames,
        'dataset_name': dataset_name,
        'time_training': time_training
    }

def evaluate_fasterrisk(clf, X_test, Y_test, time_training):
    start_test = time.time()
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_prob(X_test)
    time_testing = time.time() - start_test

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

    auc_score = roc_auc_score(Y_test, y_proba) if len(set(Y_test)) > 1 else None

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