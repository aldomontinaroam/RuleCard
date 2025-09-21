from feature_importance import get_feature_importances
import pandas as pd
import time

def time_feature_importances(
    methods: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    top_k: int = 10
) -> pd.DataFrame:
    results = []

    for method in methods:
        print(f"Running: {method}")
        try:
            start_time = time.time()
            _ = get_feature_importances(method, X, y, top_k=top_k)
            elapsed_time = time.time() - start_time

            results.append({
                "method": method,
                "time_sec": round(elapsed_time, 3)
            })

        except Exception as e:
            print(f"Error with method {method}: {e}")
            results.append({
                "method": method,
                "time_sec": None
            })

    return pd.DataFrame(results)