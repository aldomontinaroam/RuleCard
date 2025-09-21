from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import seaborn as sns
import pandas as pd

X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
max_depth = 2
lr = 1

# feature importance extraction
fi_extractor = DecisionTreeClassifier()
fi_extractor.fit(X_train, y_train)
index_feat = np.argsort(fi_extractor.feature_importances_)[::-1]
feature_names = fi_extractor.feature_names_in_

# baseline
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
y_pred_train_i = dummy.predict_proba(X_train)[:, 1]
y_pred_i = dummy.predict_proba(X_test)[:, 1]

print("Baseline AUC", roc_auc_score(y_test, y_pred_i))
print("Baseline LogLoss", log_loss(y_test, y_pred_i))

# INITIAL RESIDUAL
res_train = y_train - y_pred_train_i

# BOOSTING
trees = []
metrics = {}
res_list = [] # each entry contains a vector of the residuals of each sample
                # for that iteration, each row shows how much error remains to be corrected after the first tree
for it in range(len(index_feat)):
    i = index_feat[it]
    print(f"Iter {it}, Feature #{i} - {feature_names[i]}")

    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X_train.iloc[:, [i]], res_train)
    trees.append((model, i))

    pred_train = model.predict(X_train.iloc[:, [i]])
    pred_test = model.predict(X_test.iloc[:, [i]])

    y_pred_train_i += lr * pred_train
    y_pred_i += lr * pred_test

    y_pred_train_i_clip = np.clip(y_pred_train_i, 0, 1)
    y_pred_i_clip = np.clip(y_pred_i, 0, 1)

    auc = roc_auc_score(y_test, y_pred_i_clip)
    loss = log_loss(y_test, y_pred_i_clip)
    metrics[f"{it}"] = {
        'auc_train': roc_auc_score(y_train, y_pred_train_i_clip),
        'auc_test': auc,
        'logloss_train': log_loss(y_train, y_pred_train_i_clip),
        'logloss_test': loss
    }

    print("AUC", round(auc, 3), "LogLoss", round(loss, 4))
    
    # residual update
    res_train = y_train - y_pred_train_i_clip
    res_list.append(res_train)



def plot_metrics(metrics, res_list, figsize=(15, 4), save_path=None, dpi=400):
    
    data = []
    for epoch, vals in metrics.items():
        data.append({'Epoch': epoch, 'Metric': 'AUC', 'Set': 'Train', 'Value': vals['auc_train']})
        data.append({'Epoch': epoch, 'Metric': 'AUC', 'Set': 'Test',  'Value': vals['auc_test']})
        data.append({'Epoch': epoch, 'Metric': 'Log-loss', 'Set': 'Train', 'Value': vals['logloss_train']})
        data.append({'Epoch': epoch, 'Metric': 'Log-loss', 'Set': 'Test',  'Value': vals['logloss_test']})
    df = pd.DataFrame(data)

    sns.set_theme(style="whitegrid", font_scale=1.2, rc={"lines.linewidth": 2})
    palette = sns.color_palette("colorblind")[2:4]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    sns.lineplot(
        data=df[df['Metric'] == 'AUC'],
        x='Epoch', y='Value', hue='Set', marker='o', ax=axes[0], palette=palette
    )
    axes[0].set_title('AUC over Epochs')
    axes[0].set_ylabel('AUC')
    axes[0].set_xticks(df['Epoch'].unique()[::2])
    axes[0].legend(title='')

    sns.lineplot(
        data=df[df['Metric'] == 'Log-loss'],
        x='Epoch', y='Value', hue='Set', marker='s', ax=axes[1], palette=palette
    )
    axes[1].set_title('Log-loss over Epochs')
    axes[1].set_ylabel('Log-loss')
    axes[1].set_xticks(df['Epoch'].unique()[::2])
    axes[1].legend(title='')

    mean_residuals = [np.mean(res) for res in res_list]
    axes[2].plot(range(len(mean_residuals)), mean_residuals, marker='o', linewidth=2)
    axes[2].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[2].set_title("Residual Convergence")
    axes[2].set_xlabel("Boosting Step")
    axes[2].set_ylabel("Mean Residual")
    axes[2].grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        sns.despine(ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figura salvata in: {save_path}")

    plt.savefig("imgs/01_metrics.png", dpi=400)

plot_metrics(metrics, res_list, figsize=(15, 4))

from sklearn.tree import plot_tree

def plot_boosting_tree(model, feature_names, tree_index=0, max_depth=None, figsize=(6, 3), fontsize=10, precision=2):
    fig, ax = plt.subplots(figsize=figsize)

    plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        rounded=True,       
        impurity=False,     
        proportion=True,    
        precision=precision,
        max_depth=max_depth,
        ax=ax,
        fontsize=fontsize
    )

    ax.set_title(f"T{tree_index}: {feature_names[0]}", fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig(f"imgs/01_tree_{tree_index}_{feature_names[0]}.png", dpi=400)

for k, (model, feat_idx) in enumerate(trees[:3]):
    plot_boosting_tree(model, feature_names=[feature_names[feat_idx]], tree_index=k)

# ========================================================================================

from scipy.special import expit
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from scipy.special import expit

def run_dummy(
    X_train, X_test, y_train, y_test,
    feature_order,
    max_depth=2, lr=1.0
):
    dummy = DummyClassifier(strategy='prior')
    dummy.fit(X_train, y_train)

    y_pred_train = dummy.predict_proba(X_train)[:, 1]
    y_pred_test = dummy.predict_proba(X_test)[:, 1]

    res_train = y_train - y_pred_train
    res_list = [res_train.copy()]
    metrics = {}

    for it, i in enumerate(feature_order):
        model = DecisionTreeRegressor(max_depth=max_depth)
        model.fit(X_train.iloc[:, [i]], res_train)

        pred_train = model.predict(X_train.iloc[:, [i]])
        pred_test = model.predict(X_test.iloc[:, [i]])

        y_pred_train += lr * pred_train
        y_pred_test += lr * pred_test

        y_pred_train_clip = np.clip(y_pred_train, 0, 1)
        y_pred_test_clip = np.clip(y_pred_test, 0, 1)

        res_train = y_train - y_pred_train_clip
        res_list.append(res_train)

        metrics[it] = {
            "auc_train": roc_auc_score(y_train, y_pred_train_clip),
            "auc_test": roc_auc_score(y_test, y_pred_test_clip),
            "logloss_train": log_loss(y_train, y_pred_train_clip),
            "logloss_test": log_loss(y_test, y_pred_test_clip)
        }

    return metrics, res_list, y_pred_test_clip


def run_prior(X_train, X_test, y_train, y_test, feature_order, max_depth=2, lr=1.0):
    p0 = np.mean(y_train)
    log_odds_p0 = np.log(p0 / (1 - p0))
    
    F_train = np.full(len(y_train), log_odds_p0)
    F_test = np.full(len(y_test), log_odds_p0)

    proba_train = expit(F_train)
    proba_test = expit(F_test)
    
    res_train = y_train - proba_train
    res_list = [res_train.copy()]
    metrics = {}

    for it, i in enumerate(feature_order):
        model = DecisionTreeRegressor(max_depth=max_depth)
        model.fit(X_train.iloc[:, [i]], res_train)

        pred_train = model.predict(X_train.iloc[:, [i]])
        pred_test = model.predict(X_test.iloc[:, [i]])

        F_train += lr * pred_train
        F_test += lr * pred_test

        proba_train = expit(F_train)
        proba_test = expit(F_test)

        res_train = y_train - proba_train
        res_list.append(res_train)

        metrics[it] = {
            "logloss_train": log_loss(y_train, proba_train),
            "logloss_test": log_loss(y_test, proba_test),
            "auc_train": roc_auc_score(y_train, proba_train),
            "auc_test": roc_auc_score(y_test, proba_test)
        }

    return metrics, res_list, proba_test

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from scipy.stats import wilcoxon
from utils import load_preprocess, get_available_datasets

results = []
wilcoxon_results = []
shape = {}
class_balance = {}

n_splits = 5
seeds = [1, 2, 3, 4, 5]

custom_datasets = get_available_datasets()

for name, info in tqdm(custom_datasets.items(), desc="Total: "):
    _data = load_preprocess(name, var_threshold=0.001)
    X, y = _data['X'], _data['y']
    feature_names = _data['feature_names']
    shape[name] = X.shape
    class_balance[name] = round(y.value_counts(normalize=True), 2).to_dict()

    auc_prior_all_seeds = []
    logloss_prior_all_seeds = []
    auc_dummy_all_seeds = []
    logloss_dummy_all_seeds = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        auc_prior_list = []
        logloss_prior_list = []
        auc_dummy_list = []
        logloss_dummy_list = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            clf = DecisionTreeClassifier(random_state=seed)  # Imposta anche qui il seed
            clf.fit(X_train, y_train)
            feat_order = np.argsort(clf.feature_importances_)[::-1]

            metrics_prior, _, _ = run_prior(X_train, X_test, y_train, y_test, feat_order)
            metrics_dummy, _, _ = run_dummy(X_train, X_test, y_train, y_test, feat_order)

            last_it_prior = max(metrics_prior)
            last_it_dummy = max(metrics_dummy)

            auc_prior_list.append(metrics_prior[last_it_prior]["auc_test"])
            logloss_prior_list.append(metrics_prior[last_it_prior]["logloss_test"])
            auc_dummy_list.append(metrics_dummy[last_it_dummy]["auc_test"])
            logloss_dummy_list.append(metrics_dummy[last_it_dummy]["logloss_test"])

        auc_prior_all_seeds.extend(auc_prior_list)
        logloss_prior_all_seeds.extend(logloss_prior_list)
        auc_dummy_all_seeds.extend(auc_dummy_list)
        logloss_dummy_all_seeds.extend(logloss_dummy_list)


    # Calcola media per ciascuna metrica
    auc_prior_mean = np.mean(auc_prior_all_seeds)
    logloss_prior_mean = np.mean(logloss_prior_all_seeds)
    auc_dummy_mean = np.mean(auc_dummy_all_seeds)
    logloss_dummy_mean = np.mean(logloss_dummy_all_seeds)


    # Normalizza logloss tra prior e dummy per renderli confrontabili (min-max)
    logloss_min = min(logloss_prior_mean, logloss_dummy_mean)
    logloss_max = max(logloss_prior_mean, logloss_dummy_mean)
    if logloss_max - logloss_min == 0:
        logloss_prior_norm = 0.5
        logloss_dummy_norm = 0.5
    else:
        logloss_prior_norm = (logloss_prior_mean - logloss_min) / (logloss_max - logloss_min)
        logloss_dummy_norm = (logloss_dummy_mean - logloss_min) / (logloss_max - logloss_min)

    # Crea indicatore composito (AUC - logloss normalizzata)
    combined_prior = auc_prior_mean - logloss_prior_norm
    combined_dummy = auc_dummy_mean - logloss_dummy_norm

    try:
        stat_combined, p_combined = wilcoxon(
            [a - l for a, l in zip(auc_prior_all_seeds, logloss_prior_all_seeds)],
            [a - l for a, l in zip(auc_dummy_all_seeds, logloss_dummy_all_seeds)],
            alternative='greater'
        )

    except ValueError:
        stat_combined, p_combined = np.nan, np.nan

    results.append({
        "dataset": name,
        "shape": shape[name],
        "class_balance": class_balance[name],
        "auc_dummy_mean": auc_dummy_mean,
        "logloss_dummy_mean": logloss_dummy_mean,
        "auc_prior_mean": auc_prior_mean,
        "logloss_prior_mean": logloss_prior_mean,
        "auc_dummy_std": np.std(auc_dummy_list),
        "logloss_dummy_std": np.std(logloss_dummy_list),
        "auc_prior_std": np.std(auc_prior_list),
        "logloss_prior_std": np.std(logloss_prior_list),
        "combined_prior": combined_prior,
        "combined_dummy": combined_dummy
    })

    wilcoxon_results.append({
        "dataset": name,
        "wilcoxon_combined_p": p_combined,
        "significant_combined": p_combined < 0.05 if not np.isnan(p_combined) else False
    })

# 13 min

pd.DataFrame(results).to_csv("tables/01_results_boosting_prior_vs_dummy.csv", index=False)
# percentage of datasets where prior is significantly better than dummy
df_wilcoxon = pd.DataFrame(wilcoxon_results)
print(round(df_wilcoxon['significant_combined'].value_counts(normalize=True) * 100))
df_wilcoxon.to_csv("tables/01_wilcoxon_boosting_prior_vs_dummy.csv", index=False)