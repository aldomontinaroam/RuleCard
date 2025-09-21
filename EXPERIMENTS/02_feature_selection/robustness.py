from feature_importance import get_feature_importances
from typing import Optional, Union, List, Dict
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

def ranking_robustness_experiment_relative(
    method: str,
    X: pd.DataFrame,
    y: pd.Series,
    top_k: Optional[int] = None,
    verbose: bool = True,
    plot: bool = False,
    alpha: float = 0.05
) -> pd.DataFrame:
    base_imp = get_feature_importances(method, X, y, top_k=top_k)
    base_rank = base_imp.rank(ascending=False, method="average")
    order = base_rank.sort_values().index.tolist()

    removed: List[str] = []
    rows: List[Dict[str, Union[int, float, str]]] = []

    for i, feat in enumerate(order[:-2]):
        removed.append(feat)
        remaining_features = [c for c in X.columns if c not in removed]
        X_reduced = X[remaining_features]

        current_imp = get_feature_importances(method, X_reduced, y, top_k=top_k)
        cur_rank = current_imp.rank(ascending=False, method="average")

        base_reduced = base_rank.drop(index=removed)

        common_features = base_reduced.index.intersection(cur_rank.index)
        if len(common_features) < 2:
            break

        base_reduced_aligned = base_reduced.loc[common_features]
        cur_aligned = cur_rank.loc[common_features]

        tau, p_value = kendalltau(base_reduced_aligned, cur_aligned)
        #print(f"Step {i + 1}, tau: {tau:.3f}, p-value: {p_value:.3f}, significant: {p_value < alpha}")

        if verbose:
            print(f"\nStep {i + 1} — Feature rimossa: {feat}")
            print(f"Tau: {tau:.3f}, p-value: {p_value:.3g}")
            print("Base order:", list(base_reduced_aligned.sort_values().index))
            print("Current order:", list(cur_aligned.sort_values().index))

        rows.append({
            "step": i + 1,
            "feature_removed": feat,
            "tau": round(tau, 3) if not np.isnan(tau) else np.nan,
            "p_value": round(p_value, 3) if not np.isnan(p_value) else np.nan,
            "significant": p_value < alpha if not np.isnan(p_value) else False,
            "features_remaining": len(cur_aligned),
        })

    df_summary = pd.DataFrame(rows)

    if plot and not df_summary.empty:
        plt.figure(figsize=(6, 4))
        plt.plot(df_summary["features_remaining"], df_summary["tau"], marker="o", label="Tau")
        plt.axhline(0, color='gray', linestyle='--')
        plt.gca().invert_xaxis()
        plt.title("Kendall tau vs Number of Remaining Features")
        plt.xlabel("Number of remaining features")
        plt.ylabel("Kendall tau")
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(df_summary["features_remaining"], df_summary["p_value"], marker="x", color='red', label="p-value")
        plt.axhline(alpha, color='gray', linestyle='--', label=f"Alpha threshold = {alpha}")
        plt.gca().invert_xaxis()
        plt.title("Kendall p-value vs Number of Remaining Features")
        plt.xlabel("Number of remaining features")
        plt.ylabel("p-value")
        plt.grid(True)
        plt.legend()
        plt.show()

    return df_summary