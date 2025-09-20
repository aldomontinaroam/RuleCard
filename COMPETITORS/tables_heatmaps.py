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
pd.DataFrame(auc_vals).to_csv("csv_files/auc_all_models.csv", index_label="Dataset")

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
results.to_csv("csv_files/mean_results_all_models.csv", index=False)