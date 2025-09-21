import os
import time
import json
import traceback
import platform
import logging
from datetime import datetime
import pandas as pd
from models.slim import cv_slim_path, load_processed

# LOG
os.makedirs("logs", exist_ok=True)
LOG_PATH = os.path.join("logs", "gurobi_vs_cplex.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# CONFIG
DATASETS_CFG = {
    "adult": {"target": "Over50K"},
    "breastcancer": {"target": "Benign"},
    "bankruptcy": {"target": "NotBankrupt"},
    "haberman": {"target": "AliveAfter5Yrs"},
    "mammo": {"target": "Malignant"},
    "heart": {"target": "Disease"},
    "mushroom": {"target": "poisonous"},
    "spambase": {"target": "Spam"},
}

# ref: Table 6, Ustun & Rudin 2016
PAPER_TARGET = {
    "adult":         {"test_err_mean": 17.4, "test_err_sd": 1.4, "model_size": 18, "size_range": (7, 26)},
    "breastcancer":  {"test_err_mean":  3.4, "test_err_sd": 2.0, "model_size":  2, "size_range": (2, 2)},
    "bankruptcy":    {"test_err_mean":  0.8, "test_err_sd": 1.7, "model_size":  3, "size_range": (2, 3)},
    "haberman":      {"test_err_mean":  29.2, "test_err_sd": 14.0, "model_size":  3, "size_range": (2, 3)},
    "mammo":         {"test_err_mean":  19.5, "test_err_sd": 3.0, "model_size":  9, "size_range": (9, 11)},
    "heart":         {"test_err_mean": 16.5, "test_err_sd": 7.8, "model_size":  3, "size_range": (3, 3)},
    "mushroom":      {"test_err_mean":  0.0, "test_err_sd": 0.0, "model_size":  7, "size_range": (7, 7)},
    "spambase":      {"test_err_mean":  6.3, "test_err_sd": 1.2, "model_size": 34, "size_range": (28, 40)},
}

VERBOSE_SOLVER = 0
THREADS = None
TIME_LIMIT = 120

def save_text_safely(path, text):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Log file: {path}")
    except Exception as e:
        logger.error(f"Error log {path}: {e}")
        logger.debug(traceback.format_exc())

def save_json_safely(path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON saved: {path}")
    except Exception as e:
        logger.error(f"Error JSON {path}: {e}")
        logger.debug(traceback.format_exc())

def summarize_env():
    logger.info("=== ENV DETAILS ===")
    try:
        import multiprocessing
        cpu = multiprocessing.cpu_count()
    except Exception:
        cpu = "n/a"
    logger.info(f"OS: {platform.system()} {platform.release()} | Python: {platform.python_version()}")
    logger.info(f"CPU count: {cpu} | THREADS: {THREADS or 'auto'}")
    logger.info(f"TIME_LIMIT: {TIME_LIMIT}s | SLIM_SOLVER_VERBOSE: {VERBOSE_SOLVER}")
    try:
        import gurobipy as gp
        logger.info(f"Gurobi version: {gp.gurobi.version() if hasattr(gp,'gurobi') else gp.__version__}")
    except Exception as e:
        logger.info(e)

def run_one_dataset(name):
    t0 = time.perf_counter()

    X, Y, X_names = load_processed(name, target_col=DATASETS_CFG[name]["target"])

    N, P = X.shape
    pos = int((Y == 1).sum())
    neg = int((Y == -1).sum())
    logger.info(f"[{name}] Shape X: {X.shape} (incl. Intercept) | y:+1={pos}, -1={neg}")
    # C0 grid del paper:
    #c0_grid = [0.01, 0.075, 0.05, 0.025, 0.001, 0.9/(N*P)]
    c0_grid = [0.01, 0.05, 0.9/(N*P)]
    logger.info(f"[{name}] C0 grid: {c0_grid}")

    t_cv = time.perf_counter()
    res = cv_slim_path(
        X, Y, X_names, Y_name=DATASETS_CFG[name]["target"],
        dataset_name=name,
        C0_grid=c0_grid,
        n_splits=3,
        time_limit=TIME_LIMIT,
        threads=THREADS,
        verbose=VERBOSE_SOLVER
    )
    t_cv_done = time.perf_counter()
    logger.info(f"[{name}] cv_slim_path completato in {t_cv_done - t_cv:.1f}s")

    paper = PAPER_TARGET.get(name)
    mean_pct = round(res["cv_test_error_mean"] * 100.0, 3)
    sd_pct   = round(res["cv_test_error_std"]  * 100.0, 3)

    row = {
        "dataset": name,
        "best_C0": res["best_C0"],
        "cv_test_error_mean(%)": mean_pct,
        "cv_test_error_sd(%)": sd_pct,
        "final_model_size": int(res["final_model_size"]),
        "cv_model_size_range": f"{res['cv_model_size_range'][0]}–{res['cv_model_size_range'][1]}",
    }

    if paper:
        row.update({
            "paper_test_err_mean(%)": paper["test_err_mean"],
            "paper_test_err_sd(%)": paper["test_err_sd"],
            "paper_model_size": paper["model_size"],
            "paper_size_range": f"{paper['size_range'][0]}–{paper['size_range'][1]}",
            "Δerr_mean(pp)": round(mean_pct - paper["test_err_mean"], 3),
            "Δmodel_size": int(row["final_model_size"] - paper["model_size"]),
        })

    artifacts = {
        "final_string_model": res.get("final_string_model", ""),
        "summary": {
            "best_C0": res["best_C0"],
            "cv_test_error_mean(%)": mean_pct,
            "cv_test_error_sd(%)": sd_pct,
            "final_model_size": int(res["final_model_size"]),
            "cv_model_size_range": [int(res['cv_model_size_range'][0]), int(res['cv_model_size_range'][1])],
            "N": int(N), "P": int(P), "pos": int(pos), "neg": int(neg),
            "time_seconds_total": round(time.perf_counter() - t0, 1),
            "time_seconds_cv_path": round(t_cv_done - t_cv, 1),
        }
    }

    logger.info(f"[{name}] Best C0: {row['best_C0']} | CV test err: {mean_pct} ± {sd_pct} | "
                f"Model size (final): {row['final_model_size']} | Range CV: {row['cv_model_size_range']}")

    return row, artifacts

def main():
    summarize_env()
    os.makedirs("artifacts", exist_ok=True)

    rows = []
    datasets = ["adult",
                "bankruptcy",
                "breastcancer",
                "haberman",
                "heart",
                "mammo",
                "mushroom",
                "spambase"
                ]
    logger.info(f"Datasets: {datasets}")

    for name in datasets:
        logger.info(f"\n=== START {name.upper()} ===")
        try:
            row, artifacts = run_one_dataset(name)
            rows.append(row)

            save_text_safely(f"slim_{name}_scorecard.txt", artifacts["final_string_model"])
            save_json_safely(os.path.join("artifacts", f"slim_{name}_summary.json"), artifacts)

        except FileNotFoundError as e:
            logger.warning(f"[SKIP] {name}: file not found: {e}")
            logger.debug(traceback.format_exc())
        except AssertionError as e:
            logger.error(f"[ASSERT] {name}: {e}")
            logger.debug(traceback.format_exc())
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt.")
            break
        except Exception as e:
            logger.error(f"[ERR] {name}: {e}")
            logger.debug(traceback.format_exc())
        finally:
            logger.info(f"=== END {name.upper()} ===\n")

    if not rows:
        logger.error("No results.")
        return

    out_df = pd.DataFrame(rows)

    keep_cols = [
        "dataset","best_C0",
        "cv_test_error_mean(%)","cv_test_error_sd(%)","final_model_size","cv_model_size_range",
        "paper_test_err_mean(%)","paper_test_err_sd(%)","paper_model_size","paper_size_range",
        "Δerr_mean(pp)","Δmodel_size"
    ]
    out_df = out_df[[c for c in keep_cols if c in out_df.columns]]

    logger.info("\n=== RESULTS (Gurobi SLIM vs CPLEX SLIM) ===")
    try:
        print(out_df.to_string(index=False))
    except Exception as e:
        logger.debug(e)

    try:
        out_df.to_csv("slim_vs_paper_results.csv", index=False)
        logger.info("Saved: slim_vs_paper_results.csv")
        logger.info("Scorecard: slim_<dataset>_scorecard.txt | JSON in artifacts/")
    except Exception as e:
        logger.error(f"Error CSV: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    logger.info(f"==== GUROBI SLIM vs CPLEX SLIM | {datetime.now().isoformat()} ====")
    try:
        main()
    except Exception as e:
        logger.critical(f"Error: {e}")
        logger.debug(traceback.format_exc())
    finally:
        logger.info("==== END ====")
