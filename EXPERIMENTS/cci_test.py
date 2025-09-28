"""
Classification Competitiveness Index (CCI) for RULECARD
---------------------------------------

Evaluates the "competitiveness" of the RULECARD model compared to competitors
by integrating performance (AUC or F1), training time, and complexity (UFU, ARPF, RAR).

Idea
For each dataset d, define an index for RULECARD

    CCI_d = wP * RP_d + wT * RT_d + wC * RC_d

- RP_d: rank score of performance (high = better), in [0,1]
- RT_d: rank score of time (high = fast), in [0,1]
- RC_d: complexity score for RULECARD = 1 - GM(UFU, ARPF, RAR), in [0,1]
  (GM = geometric mean; lower complexity -> higher RC_d)

Rank-scores map ranks (Demšar 2006) to [0,1]:

    rank_score = 1 - (rank - 1) / (k - 1)

with k = number of classifiers in the dataset.

Comparisons
- For each dataset, also compute a composite index for each competitor (no complexity):

    CCI*_m = wP * RP_m + wT * RT_m

- Compute RULECARD deltas vs competitor summary statistics (mean/median/75th percentile/max).
- Compute win rates (how often RULECARD ≥ median or 75th percentile of competitors).

Default weights: wP=0.5, wT=0.2, wC=0.3 (performance first; complexity weighs more than time).

I/O expected
- 'dataset_name', 'classifier_name'
- 'AUC (%)' or 'F1 Score (%)'
- 'Time Training (s)'
- for RULECARD: 'Unique Features Utilization (UFU)', 'Active Rules per Feature (ARPF)', 'Activation Ratio (RAR)'.
  If missing, RC_d remains NaN and a warning is issued; summary statistics ignore NaNs.

Output
- CSV per dataset with CCI and components
- CSV with global summary and win rates
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, norm


def _to_numeric_clean(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )
    return pd.to_numeric(s, errors="coerce")


def _rank_score(values: pd.Series, higher_better: bool) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(1.0, index=values.index)
    ranks = values.rank(method="average", ascending=not higher_better)
    return 1.0 - (ranks - 1.0) / (len(values) - 1.0)


def _geom_mean(arr: np.ndarray, eps: float = 1e-12) -> float:
    if np.any(np.isnan(arr)):
        return np.nan
    arr = np.clip(arr, eps, 1.0)
    return float(np.exp(np.mean(np.log(arr))))


def compute_cci_vs_control(
    df_all: pd.DataFrame,
    control_name: str = "RULECARD",
    perf_metric: str = "AUC (%)",
    time_metric: str = "Time Training (s)",
    ufu_col: str = "Unique Features Utilization (UFU)",
    arpf_col: str = "Active Rules per Feature (ARPF)",
    rar_col: str = "Activation Ratio (RAR)",
    weights: tuple[float, float, float] = (0.5, 0.2, 0.3),
    outdir: str | None = None,
    make_plots: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    wP, wT, wC = weights
    assert abs(wP + wT + wC - 1.0) < 1e-9, "Weights must sum to 1.0"

    needed = {"dataset_name", "classifier_name", perf_metric, time_metric}
    missing = [c for c in needed if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing columns in df_all: {missing}")

    for c in [perf_metric, time_metric, ufu_col, arpf_col, rar_col]:
        if c in df_all.columns:
            df_all[c] = _to_numeric_clean(df_all[c])

    df = df_all.copy()
    df["_clf_up"] = df["classifier_name"].astype(str).str.upper()
    ctrl_mask = df["_clf_up"].str.fullmatch(str(control_name).upper())

    if not ctrl_mask.any():
        raise ValueError(f"Control '{control_name}' not found in 'classifier_name'.")

    records = []

    for ds, g in df.groupby("dataset_name", sort=True):
        g = g.dropna(subset=[perf_metric, time_metric], how="any").copy()
        if g.empty or (g["classifier_name"].nunique() < 2):
            continue

        perf_scores = _rank_score(g[perf_metric], higher_better=True)
        time_scores = _rank_score(g[time_metric], higher_better=True)

        g_ctrl = g[ctrl_mask.loc[g.index]]
        if g_ctrl.empty:
            continue
        ctrl_perf = float(perf_scores.loc[g_ctrl.index].mean())
        ctrl_time = float(time_scores.loc[g_ctrl.index].mean())

        if all(c in g_ctrl.columns for c in (ufu_col, arpf_col, rar_col)):
            ctrl_ufu = float(g_ctrl[ufu_col].mean())
            ctrl_arpf = float(g_ctrl[arpf_col].mean())
            ctrl_rar = float(g_ctrl[rar_col].mean())
            gm = _geom_mean(np.array([ctrl_ufu, ctrl_arpf, ctrl_rar], dtype=float))
            ctrl_comp = float(1.0 - gm) if not np.isnan(gm) else np.nan
        else:
            ctrl_comp = np.nan
            warnings.warn(f"[{ds}] Missing complexity columns for '{control_name}'. RC_d set to NaN.")

        ctrl_cci = wP * ctrl_perf + wT * ctrl_time + wC * ctrl_comp

        comp_mask = ~g.index.isin(g_ctrl.index)
        comp_perf = perf_scores.loc[comp_mask]
        comp_time = time_scores.loc[comp_mask]
        comp_cci_star = wP * comp_perf + wT * comp_time

        comp_mean = float(comp_cci_star.mean())
        comp_median = float(comp_cci_star.median())
        comp_p75 = float(comp_cci_star.quantile(0.75))
        comp_max = float(comp_cci_star.max())

        records.append({
            "dataset_name": ds,
            "k_classifiers": int(g["classifier_name"].nunique()),
            "CCI_rulecard": float(ctrl_cci),
            "RP_perf_rulecard": ctrl_perf,
            "RT_time_rulecard": ctrl_time,
            "RC_complexity_rulecard": ctrl_comp,
            "CCI_minus_mean": float(ctrl_cci - comp_mean) if np.isfinite(ctrl_cci) else np.nan,
            "CCI_minus_median": float(ctrl_cci - comp_median) if np.isfinite(ctrl_cci) else np.nan,
            "CCI_minus_p75": float(ctrl_cci - comp_p75) if np.isfinite(ctrl_cci) else np.nan,
            "CCI_minus_max": float(ctrl_cci - comp_max) if np.isfinite(ctrl_cci) else np.nan,
        })

    if not records:
        raise ValueError("No valid datasets found after processing.")

    df_ds = pd.DataFrame.from_records(records).sort_values("dataset_name").reset_index(drop=True)

    v_med = df_ds["CCI_minus_median"].dropna()
    v_p75 = df_ds["CCI_minus_p75"].dropna()
    v_max = df_ds["CCI_minus_max"].dropna()

    wins_median = float((v_med >= 0).mean()) if len(v_med) else np.nan
    wins_p75 = float((v_p75 >= 0).mean()) if len(v_p75) else np.nan
    wins_max = float((v_max >= 0).mean()) if len(v_max) else np.nan

    df_summary = pd.DataFrame([
        {"metric": "datasets", "value": int(df_ds.shape[0])},
        {"metric": "CCI_rulecard_mean", "value": float(np.nanmean(df_ds["CCI_rulecard"]))},
        {"metric": "CCI_rulecard_std", "value": float(np.nanstd(df_ds["CCI_rulecard"], ddof=0))},
        {"metric": "winrate_vs_median", "value": wins_median},
        {"metric": "winrate_vs_p75", "value": wins_p75},
        {"metric": "winrate_vs_max", "value": wins_max},
        {"metric": "weights", "value": str(weights)},
        {"metric": "perf_metric", "value": perf_metric},
        {"metric": "time_metric", "value": time_metric},
        {"metric": "control", "value": control_name},
    ])

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        df_ds.to_csv(os.path.join(outdir, f"CCI_vs_{control_name.upper()}.csv"), index=False)
        df_summary.to_csv(os.path.join(outdir, f"CCI_summary_{control_name.upper()}.csv"), index=False)

    if outdir and make_plots:
        df_plot = df_ds.dropna(subset=["CCI_rulecard"])  
        fig = plt.figure(figsize=(10, max(4, 0.25 * len(df_plot))))
        ax = fig.gca()
        ax.barh(df_plot["dataset_name"], df_plot["CCI_rulecard"])
        ax.axvline(0.0, linestyle="--")
        ax.set_xlabel("CCI (0..1)")
        ax.set_title(f"CCI of {control_name} per dataset")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"CCI_{control_name.upper()}_per_dataset.png"), dpi=300)
        plt.close(fig)

    return df_ds, df_summary


def wilcoxon_delta(
    df_ds: pd.DataFrame,
    delta_col: str = "CCI_minus_median",
    alternative: str = "two-sided",
) -> pd.DataFrame:
    if delta_col not in df_ds.columns:
        raise ValueError(f"Column '{delta_col}' not found in df_ds")
    x = df_ds[delta_col].astype(float).dropna()
    x = x[np.abs(x) > 0]
    N = int(x.size)
    if N == 0:
        return pd.DataFrame([{ "N": 0, "mean_delta": np.nan, "median_delta": np.nan, "W": np.nan, "p_value": 1.0, "Z": 0.0, "r": np.nan }])
    res = wilcoxon(x, alternative=alternative, zero_method="wilcox")
    p = float(res.pvalue)
    z_abs = float(norm.isf(p / 2.0)) if p > 0 else np.inf
    sign = float(np.sign(np.median(x))) or 1.0
    Z = sign * z_abs
    r = float(Z / np.sqrt(N)) if np.isfinite(Z) else np.nan
    return pd.DataFrame([{ "N": N, "mean_delta": float(np.mean(x)), "median_delta": float(np.median(x)), "W": float(res.statistic), "p_value": p, "Z": Z, "r": r }])


def weight_sensitivity_grid(
    df_all: pd.DataFrame,
    control_name: str = "RULECARD",
    perf_metric: str = "AUC (%)",
    time_metric: str = "Time Training (s)",
    ufu_col: str = "Unique Features Utilization (UFU)",
    arpf_col: str = "Active Rules per Feature (ARPF)",
    rar_col: str = "Activation Ratio (RAR)",
    wP_range: tuple[float, float, float] = (0.4, 0.6, 0.05),
    wT_range: tuple[float, float, float] = (0.1, 0.3, 0.05),
    outdir: str | None = None,
    fname_prefix: str = "CCI_weight_sensitivity",
) -> pd.DataFrame:
    wP_vals = np.round(np.arange(wP_range[0], wP_range[1] + 1e-9, wP_range[2]), 10)
    wT_vals = np.round(np.arange(wT_range[0], wT_range[1] + 1e-9, wT_range[2]), 10)
    rows = []
    for wP in wP_vals:
        for wT in wT_vals:
            wC = float(1.0 - wP - wT)
            if wC < 0.0 or wC > 1.0:
                continue
            try:
                _, df_sum = compute_cci_vs_control(
                    df_all,
                    control_name=control_name,
                    perf_metric=perf_metric,
                    time_metric=time_metric,
                    ufu_col=ufu_col,
                    arpf_col=arpf_col,
                    rar_col=rar_col,
                    weights=(float(wP), float(wT), float(wC)),
                    outdir=None,
                    make_plots=False,
                )
                summary = dict(zip(df_sum["metric"].astype(str), df_sum["value"]))
                rows.append({
                    "wP": float(wP),
                    "wT": float(wT),
                    "wC": float(wC),
                    "winrate_vs_median": float(summary.get("winrate_vs_median", np.nan)),
                    "winrate_vs_p75": float(summary.get("winrate_vs_p75", np.nan)),
                    "winrate_vs_max": float(summary.get("winrate_vs_max", np.nan)),
                    "CCI_rulecard_mean": float(summary.get("CCI_rulecard_mean", np.nan)),
                })
            except Exception as e:
                warnings.warn(f"Grid (wP={wP}, wT={wT}) failed: {e}")
    grid = pd.DataFrame(rows).sort_values(["wP", "wT"]).reset_index(drop=True)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        grid.to_csv(os.path.join(outdir, f"{fname_prefix}.csv"), index=False)
    return grid

def load_model_csvs(
    folder: str,
    required_columns: tuple[str, ...] = ("AUC (%)", "F1 Score (%)", "Time Training (s)")
) -> pd.DataFrame:
    records = []
    for fn in os.listdir(folder):
        if not fn.lower().endswith(".csv"):
            continue
        path = os.path.join(folder, fn)
        model = os.path.splitext(fn)[0]
        df = pd.read_csv(path, index_col=0)
        if df.index.name is None or str(df.index.name).startswith("Unnamed"):
            df.index.name = "dataset_name"
        df["dataset_name"] = df.index.astype(str)
        df["classifier_name"] = model.replace("_comparison", "").upper()
        records.append(df.reset_index(drop=True))
    if not records:
        raise ValueError(f"No CSV found in {folder}")
    df_all = pd.concat(records, ignore_index=True)
    missing = [c for c in required_columns if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSVs: {missing}")
    keep_cols = ["dataset_name", "classifier_name"] + list(required_columns)
    for c in ["Unique Features Utilization (UFU)", "Active Rules per Feature (ARPF)", "Activation Ratio (RAR)"]:
        if c in df_all.columns:
            keep_cols.append(c)
    df_all = df_all[keep_cols].copy()
    for c in required_columns:
        df_all[c] = _to_numeric_clean(df_all[c])
    return df_all


if __name__ == "__main__":
    FOLDER = "RESULTS"
    OUTDIR = "cci_test"
    PERF_METRIC = "AUC (%)"  # or "F1 Score (%)"
    WEIGHTS = (0.5, 0.2, 0.3)

    df_all = load_model_csvs(
        FOLDER,
        required_columns=("AUC (%)", "F1 Score (%)", "Time Training (s)")
    )

    df_cci, df_sum = compute_cci_vs_control(
        df_all,
        control_name="RULECARD",
        perf_metric=PERF_METRIC,
        time_metric="Time Training (s)",
        weights=WEIGHTS,
        outdir=OUTDIR,
        make_plots=True,
    )

    print("==== CCI per dataset (head) ====")
    print(df_cci.head())
    print("==== Summary ====")
    print(df_sum)

    wil = wilcoxon_delta(df_cci, delta_col="CCI_minus_median", alternative="two-sided")
    print("==== Wilcoxon on Δ_median ====")
    print(wil)
    if OUTDIR:
        wil.to_csv(os.path.join(OUTDIR, "CCI_wilcoxon_delta_median.csv"), index=False)

    grid = weight_sensitivity_grid(
        df_all,
        control_name="RULECARD",
        perf_metric=PERF_METRIC,
        time_metric="Time Training (s)",
        ufu_col="Unique Features Utilization (UFU)",
        arpf_col="Active Rules per Feature (ARPF)",
        rar_col="Activation Ratio (RAR)",
        wP_range=(0.4, 0.6, 0.05),
        wT_range=(0.1, 0.3, 0.05),
        outdir=OUTDIR,
        fname_prefix="CCI_weight_sensitivity",
    )
    print("==== Weight sensitivity (head) ====")
    print(grid.head())

    if not grid.empty:
        top = grid.sort_values(["winrate_vs_median","winrate_vs_p75","winrate_vs_max","CCI_rulecard_mean"], ascending=False).head(5)
        print("==== Top-5 weight combinations by win-rate vs median ====")
        print(top)
