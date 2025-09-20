import pandas as pd
import os

dfs = {}
csv_folder = "RESULTS"

for csv_file in os.listdir(csv_folder):
    if csv_file.endswith(".csv"):
        df = pd.read_csv(os.path.join(csv_folder, csv_file))
        dfs[csv_file] = df

dfs = {k.replace("_comparison.csv", ""): v for k, v in dfs.items()}
for model, df in dfs.items():
    df.rename(columns={"Unnamed: 0": "Dataset"}, inplace=True)

ebm_auc = dfs["ebm"][["Dataset", "AUC (%)"]]
ebm_lasso_auc = dfs["ebm_lasso"][["Dataset", "AUC (%)"]]
treeGAM_auc = dfs["treeGAM"][["Dataset", "AUC (%)"]]
fasterrisk_auc = dfs["fasterrisk"][["Dataset", "AUC (%)"]]
figs_auc = dfs["figs"][["Dataset", "AUC (%)"]]
slim_auc = dfs["slim"][["Dataset", "AUC (%)"]]
gam_auc = dfs["gam"][["Dataset", "AUC (%)"]]

auc_vals = {}

for model in dfs.keys():
    auc_vals[model] = {dataset: float(round(df["AUC (%)"].values[0], 2)) for dataset, df in dfs[model].groupby("Dataset")}
pd.DataFrame(auc_vals).to_csv("csv_files_img/auc_all_models.csv", index_label="Dataset")

METRICS = [
    "Accuracy (%)",
    "Precision (%)",
    "Recall (%)",
    "F1 Score (%)",
    "Balanced Accuracy (%)",
    "MCC (%)",
    "AUC (%)",
    "Time Training (s)",
]

def mean_table_by_model(dfs, metrics=METRICS, decimals=2):
    means_by_model = {
        model: (
            df.reindex(columns=metrics)
              .apply(pd.to_numeric, errors="coerce")
              .mean()
        )
        for model, df in dfs.items()
    }

    results = (
        pd.DataFrame.from_dict(means_by_model, orient="index")
          .round(decimals)
          .reset_index()
          .rename(columns={"index": "Model"})
          .loc[:, ["Model"] + metrics]
    )
    return results

results = mean_table_by_model(dfs)
results.to_csv("csv_files_img/mean_results_all_models.csv", index=False)



import os, glob, re, io, base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display, HTML

BASE_DIR = "RESULTS"
files = sorted(glob.glob(os.path.join(BASE_DIR, "*_comparison.csv")))

def norm(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", str(s)).strip().lower()

models = []
metric_tables = {"F1 Score": {}, "MCC": {}, "Balanced Accuracy": {}}

def ingest_df(df, model_name):
    if df.columns[0].lower() not in [c.lower() for c in ["Accuracy (%)","Precision (%)","Recall (%)","F1 Score (%)","Balanced Accuracy (%)","MCC (%)","Error Rate (%)","AUC (%)","Time Training (s)"]]:
        df = df.rename(columns={df.columns[0]: "dataset"}).set_index("dataset")
    else:
        for c in df.columns:
            if c.lower() == "dataset":
                df = df.set_index(c)
                break

    cols_norm = {norm(c): c for c in df.columns}
    def find_col(keywords):
        for k in cols_norm.keys():
            if all(kw in k for kw in keywords):
                return cols_norm[k]
        return None

    col_f1 = find_col(["f1"])
    col_mcc = find_col(["mcc"])
    col_ba  = find_col(["balanced", "accuracy"])

    for label, col in [("F1 Score", col_f1), ("MCC", col_mcc), ("Balanced Accuracy", col_ba)]:
        if col is None: 
            continue
        series = df[col].copy()
        if np.nanmax(series.values) <= 1.5:
            series = series * 100.0
        metric_tables[label][model_name] = series

for f in files:
    base = os.path.basename(f)
    m = re.match(r"(.+?)_comparison\.", base)
    model_name = m.group(1) if m else os.path.splitext(base)[0]
    models.append(model_name)
    df = pd.read_csv(f, sep=None, engine="python")
    ingest_df(df, model_name)

matrices = {}
for label, per_model in metric_tables.items():
    if not per_model: 
        continue
    all_datasets = sorted(set().union(*[s.index.tolist() for s in per_model.values()]))
    mat = pd.DataFrame(index=all_datasets, columns=models, dtype=float)
    for model_name, s in per_model.items():
        mat.loc[s.index, model_name] = s.values
    matrices[label] = mat

labels = list(matrices.keys())
n = len(labels)

fig, axes = plt.subplots(
    1, n,
    figsize=(7 * n, max(5, 0.35 * max(len(m.index) for m in matrices.values()))),
    constrained_layout=True
)

plt.subplots_adjust(wspace=0.4)


if n == 1:
    axes = [axes]

for ax, label in zip(axes, labels):
    mat = matrices[label]
    cax = ax.matshow(mat.T, cmap='bone', aspect='auto', vmin=0, vmax=100)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label=label + " (%)")

    ax.set_xticks(np.arange(len(mat.index)))
    ax.set_yticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.index, rotation=90)
    ax.set_yticklabels(mat.columns)

    for (i, j), val in np.ndenumerate(mat.T.values):
        if not np.isnan(val):
            ax.text(
                j, i, f"{val:.1f}",
                ha="center", va="center",
                color="white" if val < 50 else "black",
                fontsize=8
            )

    for j, dataset in enumerate(mat.index):
        col_vals = mat.loc[dataset].values.astype(float)
        if np.all(np.isnan(col_vals)):
            continue
        max_val = np.nanmax(col_vals)
        max_indices = np.where(col_vals == max_val)[0]
        for i_max in max_indices:
            rect = patches.Rectangle(
                (j - 0.5, i_max - 0.5), 1, 1,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)

    ax.set_title(label)

plt.savefig(os.path.join("csv_files_img/comparison_heatmaps_f1_ba_mcc.png"), dpi=300)