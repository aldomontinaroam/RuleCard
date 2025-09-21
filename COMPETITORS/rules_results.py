from models.ebm import run_ebm, evaluate_ebm
from models.ebm_lasso import run_ebm_lasso, evaluate_ebm_lasso
from models.figs import run_figs, evaluate_figs
from models.gam import run_gam, evaluate_gam
from models.slim import run_slim, evaluate_slim
from models.treeGAM import run_treeGAM, evaluate_treeGAM
# from models.fasterrisk import run_fasterrisk, evaluate_fasterrisk

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
import numpy as np
import matplotlib.pyplot as plt
import textwrap

DT_FOLDER = "DATA/RULES/DT/"
RF_FOLDER = "DATA/RULES/RF/"
RESULTS_FOLDER = "DATA/RULES/RESULTS/"

results = {
    "DT": {},
    "RF": {}
}

metrics = {
    "DT": {},
    "RF": {}
}

METRICS = [
    'Accuracy (%)','Precision (%)','Recall (%)','F1 Score (%)',
    'Balanced Accuracy (%)','MCC (%)','Error Rate (%)','AUC (%)','Time Training (s)'
]

run_eval_fns = [
    (run_slim, evaluate_slim),
    (run_gam, evaluate_gam),
    (run_ebm, evaluate_ebm),
    (run_ebm_lasso, evaluate_ebm_lasso),
    (run_treeGAM, evaluate_treeGAM),
    (run_figs, evaluate_figs),
    # (run_fasterrisk, evaluate_fasterrisk),  # venv issue
]

def run_model(model_name, run_fn, eval_fn, X, Y, X_names, Y_name, dataset_name):
    run_fn_name = run_fn.__name__.replace("run_", "").replace("_","").upper()

    if run_fn is run_gam:
        out = run_fn(X, Y, X_names, Y_name, dataset_name, top_k=10,
                     selector="gam_auc", verbose=False)
        best_gam = out['model']
        X_test_sel = out['X_test_selected']
        Y_test = out['Y_test']
        time_training = out['time_training']
        metrics[model_name][(dataset_name, run_fn_name)] = eval_fn(
            best_gam, X_test_sel, Y_test, time_training, dataset_name
        )

    elif run_fn is run_slim:
        Y_mapped = Y.map({0: -1, 1: 1})
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y_mapped, test_size=0.2, random_state=42, stratify=Y_mapped
        )
        results[model_name][dataset_name], evals_df, _ = run_fn(
            X_train.values, Y_train.values, X_names, Y_name,
            dataset_name=dataset_name, time_limit=120, verbose=0
        )
        metrics[model_name][(dataset_name, run_fn_name)], _ = eval_fn(
            results[model_name][dataset_name], evals_df, X_test, Y_test
        )

    elif run_fn is run_figs:
        results[model_name][dataset_name] = run_fn(
            X.values, Y.values, X_names, Y_name, dataset_name=dataset_name,
            time_limit=120, max_rules=X.shape[1], max_trees=X.shape[1]//2
        )
        metrics[model_name][(dataset_name, run_fn_name)], _ = eval_fn(
            model=results[model_name][dataset_name]["model"],
            X_test=results[model_name][dataset_name]["X_test"],
            Y_test=results[model_name][dataset_name]["Y_test"],
            time_training=results[model_name][dataset_name]["time_training"]
        )

    else:
        results[model_name][dataset_name] = run_fn(X, Y, X_names, Y_name, dataset_name)
        metrics[model_name][(dataset_name, run_fn_name)], _ = eval_fn(
            model=results[model_name][dataset_name]["model"],
            X_test=results[model_name][dataset_name]["X_test"],
            Y_test=results[model_name][dataset_name]["Y_test"],
            time_training=results[model_name][dataset_name]["time_training"]
        )

if __name__ == "__main__":
    # DECISION TREES
    for file in os.listdir(DT_FOLDER):
        print(f"\nProcessing {file}...")
        data = pd.read_csv(os.path.join(DT_FOLDER, file))
        X, y = data.drop(columns=["class"]), data["class"]
        X_names, Y_name = X.columns.tolist(), "class"
        dataset_name = file.replace("_rules.csv", "")
        for run_fn, evaluate_fn in tqdm(run_eval_fns):
            run_model("DT", run_fn, evaluate_fn, X, y, X_names, Y_name, dataset_name)

    # RANDOM FORESTS
    for file in os.listdir(RF_FOLDER):
        print(f"\nProcessing {file}")
        data = pd.read_csv(os.path.join(RF_FOLDER, file))
        X, y = data.drop(columns=["class"]), data["class"]
        X_names, Y_name = X.columns.tolist(), "class"
        dataset_name = file.replace("_rules.csv", "")
        for run_fn, evaluate_fn in tqdm(run_eval_fns):
            run_model("RF", run_fn, evaluate_fn, X, y, X_names, Y_name, dataset_name)

    rows = []
    for src in metrics:  # DT / RF
        for (dataset, model), df in metrics[src].items():
            series = pd.Series(df)
            row = series.reindex(METRICS)
            rows.append({
                "Dataset": dataset,
                "RulesFrom": src,
                "Model": model,
                **row.to_dict()
            })

    master = pd.DataFrame(rows)
    master = master[['Dataset','RulesFrom','Model'] + METRICS]
    master.sort_values(["Dataset","RulesFrom","Model"], inplace=True)
    master.to_csv(os.path.join(RESULTS_FOLDER, "MASTER_metrics.csv"), index=False)

    pivot = master.pivot_table(index=["Dataset","RulesFrom"], columns="Model",
                               values=["Accuracy (%)","AUC (%)","F1 Score (%)"])
    pivot.to_csv(os.path.join(RESULTS_FOLDER, "MASTER_pivot.csv"))

    print("Saved: MASTER_metrics.csv and MASTER_pivot.csv")

    all_files = glob.glob("RESULTS/*_comparison.csv")
    extended_rows = []
    for file in all_files:
        df = pd.read_csv(file)
        
        df["RulesFrom"] = "BASE"
        model_name = file.replace("RESULTS/", "").replace("_comparison.csv", "").replace("_","").upper()
        df["Model"] = model_name
        
        extended_rows.append(df)
    df_new = pd.concat(extended_rows, ignore_index=True)
    df_new = df_new[df_new['Model'] != 'SCORECARD'].rename(columns={"Unnamed: 0": "Dataset"})
    df_final = pd.concat([master, df_new], ignore_index=True)
    df_final.to_csv(f"{RESULTS_FOLDER}MASTER_comparison_extended_base.csv", index=False)

df = df_final.copy(deep=True)

metrics = ['Accuracy (%)', 'Precision (%)',
       'Recall (%)', 'F1 Score (%)', 'Balanced Accuracy (%)', 'MCC (%)', 
       'AUC (%)', 'Time Training (s)']

results = []

for metric in metrics:
    # Pivot: (Dataset, Model) × RulesFrom
    pivot = df.pivot_table(index=["Dataset", "Model"], 
                           columns="RulesFrom", 
                           values=metric)
    
    winners = pivot.idxmax(axis=1)
    counts = winners.value_counts(normalize=True) * 100
    
    for variant in ["BASE", "DT", "RF"]:
        results.append({
            "Metric": metric,
            "Variant": variant,
            "Win %": counts.get(variant, 0.0)
        })

df_results = pd.DataFrame(results)

pivot_plot = df_results.pivot(index="Metric", columns="Variant", values="Win %")

ax = pivot_plot.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 4),
    color=["grey", "lightgreen", "teal"],
    width=0.7
)

plt.ylabel("% of win")
plt.title("% of win per metric (BASE vs DT vs RF)")
plt.legend(title="Dataset transformation", bbox_to_anchor=(1.05, 1), loc="upper left")

labels = [textwrap.fill(label, 12) for label in pivot_plot.index]
labels = [label.replace("(%)", "") for label in labels]
ax.set_xticklabels(labels, fontsize=7, rotation=0)

plt.tight_layout()
plt.savefig(f"csv_files_img/MASTER_comparison_extended_base_win_percentage.png", dpi=300)