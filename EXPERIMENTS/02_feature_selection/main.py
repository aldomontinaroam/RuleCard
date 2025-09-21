from utils import load_preprocess, get_available_datasets, analyze_method_consistency
from typing import Literal, Dict
from FeatureSelectionEvaluator import FeatureSelectionEvaluator
import time

start_time = time.time()

available_datasets = get_available_datasets()
datasets = {}

for name, df in available_datasets.items():
    data_dict = load_preprocess(name)
    
    datasets[name] = {
        'data': data_dict['data'],
        'X': data_dict['X'],
        'y': data_dict['y'],
        'feature_names': data_dict['feature_names']
    }

methods = [
            "select_kbest", "select_kbest_mi", "select_kbest_chi2",
            "random_forest", "lightgbm",
            "logistic_regression_l2", "logistic_regression_l1",
            "xgboost", "catboost"
            #, "rfe_logistic_l2", "rfe_logistic_l1", "rfe_lgbm",
]

Metric = Literal["jaccard", "overlap"]

scores_per_dataset = []
evaluators = {}
for name, d in datasets.items():
    X, y = d['X'], d['y']
    n_records = X.shape[0]
    n_features = X.shape[1]
    dataset_dimension = n_records * n_features

    print(f"\n[{name}] Features: {n_features}, Samples: {n_records}, Dimension: {dataset_dimension}")

    evaluator = FeatureSelectionEvaluator(
        methods=methods,
        X=X,
        y=y,
        n_runs=3,
        top_k=15,
        val_size=0.3,
        seed=42,
        n_jobs=-1,
        alpha=0.05
    )
    evaluators[name] = evaluator
    results = evaluator.run()

    scores_df = evaluator.get_scores_df()

    scores_df["n_records"] = n_records
    scores_df["n_features"] = n_features
    scores_df["dimension"] = dataset_dimension

    scores_df.to_csv(f"results/feature_selection_{name}.csv", index=False)

    scores_per_dataset.append({
        "dataset": name,
        "scores_df": scores_df
    })


consistency_df = analyze_method_consistency(scores_per_dataset)
consistency_df.to_csv("results/fs_consistency.csv", index=False)

end_time = time.time()

total_time = end_time - start_time
minutes = int(total_time // 60)
seconds = int(total_time % 60)

print(f"\nTIME: {total_time:.2f} sec (circa {minutes} min {seconds} sec)")