from typing import List, Union, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.utils import resample
from joblib import Parallel, delayed
from feature_importance import get_feature_importances

def bootstrap_feature_analysis_parallel(
    method: str,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    *,
    n_runs: int = 10,
    top_k: int = 10,
    random_state: Optional[int] = None,
    n_jobs: int = -1
) -> List[pd.Series]:

    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**32 - 1, size=n_runs)

    def bootstrap_run(seed):
        indices = resample(range(len(X)), replace=True, random_state=seed)
        X_res = X.iloc[indices]
        y_res = y.iloc[indices]
        importances = get_feature_importances(method=method, X=X_res, y=y_res, top_k=top_k)
        return importances

    rankings = Parallel(n_jobs=n_jobs)(delayed(bootstrap_run)(seed) for seed in seeds)
    return rankings


def ranking_agreement(
    rankings: Sequence[pd.Series],
    *,
    metric: str = "overlap",
    k: Optional[int] = None
) -> float:

    if k is None or k <= 0:
        raise ValueError("Parameter 'k' must be specified and > 0 for agreement metrics")

    n = len(rankings)
    if n < 2:
        return 0.0

    scores = []
    for i in range(n):
        top_i = set(rankings[i].index[:k])
        for j in range(i + 1, n):
            top_j = set(rankings[j].index[:k])

            intersection = len(top_i & top_j)
            union = len(top_i | top_j)

            if metric == "jaccard":
                score = intersection / union
            else:  # overlap
                score = intersection / k

            scores.append(score)

    return float(np.mean(scores)) if scores else 0.0