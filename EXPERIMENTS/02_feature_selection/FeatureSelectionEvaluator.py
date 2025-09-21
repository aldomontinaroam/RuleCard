import pandas as pd
import numpy as np
from collections import OrderedDict
from typing import List
from tqdm import tqdm
from stability import bootstrap_feature_analysis_parallel, ranking_agreement
from robustness import ranking_robustness_experiment_relative
from performance import performance_on_topk
from time_experiment import time_feature_importances

class FeatureSelectionEvaluator:
    def __init__(
        self,
        methods: List[str],
        X: pd.DataFrame,
        y: pd.Series,
        n_runs: int = 10,
        top_k: int = 10,
        val_size: float = 0.2,
        seed: int = 0,
        n_jobs: int = -1,
        alpha: float = 0.05
    ):
        """
        Evaluation of feature selection methods based on:
        - Bootstrap stability
        - Robustness
        - Top-k performance
        - Execution time
        """
        self.methods = methods
        self.X = X
        self.y = y
        self.n_runs = n_runs
        self.top_k = top_k
        self.val_size = val_size
        self.seed = seed
        self.n_jobs = n_jobs
        self.alpha = alpha
        self.results_summary: List[OrderedDict] = []
        self.performance_weights = {
            "acc": 0.1,
            "bal_acc": 0.1,
            "recall": 0.1,
            "f1": 0.2,
            "mcc": 0.2,
            "auc": 0.1,
            "avgp": 0.2
        }

    def evaluate_method(self, method: str) -> OrderedDict:
        summary = OrderedDict()
        summary["method"] = method

        # TIME
        try:
            times = time_feature_importances(methods=[method], X=self.X, y=self.y, top_k=self.top_k)
            summary["time_sec"] = times[times["method"] == method]["time_sec"].values[0]
        except Exception as e:
            print(f"[{method}] ERROR in importance timing: {e}")
            summary["time_sec"] = np.nan

        # STABILITY
        try:
            rankings = bootstrap_feature_analysis_parallel(method=method, X=self.X, y=self.y,
                                                          n_runs=self.n_runs, top_k=self.top_k,
                                                          random_state=self.seed, n_jobs=self.n_jobs)
            summary["jaccard"] = ranking_agreement(rankings, metric="jaccard", k=self.top_k)
            summary["overlap"] = ranking_agreement(rankings, metric="overlap", k=self.top_k)
        except Exception as e:
            print(f"[{method}] ERROR in stability: {e}")
            summary["jaccard"] = np.nan
            summary["overlap"] = np.nan

        # ROBUSTNESS
        try:
            robustness_df = ranking_robustness_experiment_relative(method=method, X=self.X, y=self.y,
                                                                    top_k=self.top_k, verbose=False, plot=False, alpha=self.alpha)
            if not robustness_df.empty and "significant" in robustness_df.columns:
                tau_sig = robustness_df.loc[robustness_df["significant"], "tau"]
                mean_tau_sig = tau_sig.mean() if not tau_sig.empty else 0.0
                perc_sig = robustness_df["significant"].mean() * 100
            else:
                mean_tau_sig = 0.0
                perc_sig = 0.0

            summary["mean_tau_significant"] = mean_tau_sig
            summary["perc_significant_steps"] = perc_sig
        except Exception as e:
            print(f"[{method}] ERROR in robustness: {e}")
            summary["mean_tau_significant"] = np.nan
            summary["perc_significant_steps"] = np.nan

        # PERFORMANCE
        try:
            perf_results = performance_on_topk(method=method, X_train=self.X, y_train=self.y,
                                            val_size=self.val_size, k=self.top_k,
                                            seed=self.seed, include_all=True)
            for key, value in perf_results.items():
                summary[key] = value
        except Exception as e:
            print(f"[{method}] ERROR in performance: {e}")
            for metric in ["rf_top_acc", "rf_top_bal_acc", "rf_top_recall", "rf_top_f1",
                        "rf_top_mcc", "rf_top_auc", "rf_top_avgp",
                        "lgbm_top_acc", "lgbm_top_bal_acc", "lgbm_top_recall", "lgbm_top_f1",
                        "lgbm_top_mcc", "lgbm_top_auc", "lgbm_top_avgp"]:
                summary[metric] = np.nan

        return summary


    def run(self, verbose: bool = True) -> List[OrderedDict]:
        results = []

        iterator = tqdm(self.methods, desc="Feature Selection Methods") if verbose else self.methods

        for m in iterator:
            try:
                res = self.evaluate_method(m)
                results.append(res)
            except Exception as err:
                print(f"[{m}] ERROR: {err}")

        self.results_summary = results
        return results

    def get_results(self) -> List[OrderedDict]:
        return self.results_summary
    
    def get_scores_df(self) -> pd.DataFrame:
        score_df = pd.DataFrame(self.results_summary)
        if score_df.empty:
            return score_df
        
        score_df["stability"] = ((score_df["jaccard"] + score_df["overlap"]) / 2).round(2) # media tra jaccard e overlap
        score_df["robustness"] = (score_df["mean_tau_significant"] * (score_df["perc_significant_steps"] / 100)).round(2)

        def compute_weighted_perf(df, prefix, weights=self.performance_weights):
            return (
                df[f"{prefix}_acc"] * weights["acc"] +
                df[f"{prefix}_bal_acc"] * weights["bal_acc"] +
                df[f"{prefix}_recall"] * weights["recall"] +
                df[f"{prefix}_f1"] * weights["f1"] +
                df[f"{prefix}_mcc"] * weights["mcc"] +
                df[f"{prefix}_auc"] * weights["auc"] +
                df[f"{prefix}_avgp"] * weights["avgp"]
            )
        
        score_df["perf_rf"] = compute_weighted_perf(score_df, "rf_top")
        score_df["perf_lgbm"] = compute_weighted_perf(score_df, "lgbm_top")
        score_df["performance"] = ((score_df["perf_rf"] + score_df["perf_lgbm"]) / 2).round(2)

        score_weights = {
            "time_sec": 0.3,
            "stability": 0.2,
            "robustness": 0.2,
            "performance": 0.3
        }
        score_df["SCORE"] = (
            score_df["stability"] * score_weights["stability"] +
            score_df["robustness"] * score_weights["robustness"] +
            score_df["performance"] * score_weights["performance"] +
            1/(score_df["time_sec"]+1e-6) * score_weights["time_sec"]
        )

        score_df = score_df.sort_values("SCORE", ascending=False).reset_index(drop=True)

        return score_df[
            ["method", "SCORE", "stability", "robustness", "performance", "time_sec"]
        ]
