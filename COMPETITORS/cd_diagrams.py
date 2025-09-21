"""
Evaluation of multiple classifiers over multiple datasets with:
- Omnibus Friedman test (Iman-Davenport) on ranks per dataset
- Post-hoc:
 ->vs control: Dunn/Bonferroni in the CD-plot; table of p-values corrected with Holm
 ->all-vs-all: Nemenyi in the CD-plot; possibly MCM with Wilcoxon + Holm

REFERENCES:
- Demsar, J. (2006). Statistical comparisons of classifiers over multiple data sets. JMLR.
- Iman, R. L., & Davenport, J. M. (1980). Approximations of the critical region of the Friedman statistic. Communications in Statistics.
- Dunn, O. J. (1961). Multiple comparisons among means. Journal of the American Statistical Association.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics.
- Hochberg, Y. (1988). A sharper Bonferroni procedure. Biometrika.
- García, S., & Herrera, F. (2008). An extension on “Statistical comparisons of classifiers”. Pattern Recognition Letters.
- Bergmann, B., & Hommel, G. (1988). Improvements of general multiple test procedures. Biometrics.

KEY POINTS:
- Complete block design (each classifier evaluated on each dataset).
- Non-parametric tests on ranks: no assumption of normality/homoscedasticity.
- The CD-plots visualize Nemenyi or Dunn thresholds (a single threshold): stepwise procedures
  (e.g. Holm) are NOT representable with a single CD line; for this reason, the plots and tables
  may diverge, this is expected and should be explained.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "Arial"

# --- AEON (CD-diagram) --------------------------------------------------------
try:
    # aeon>=0.7
    from aeon.visualisation import (
        plot_critical_difference,
        create_multi_comparison_matrix,
    )
except Exception as e:
    raise RuntimeError(
        "Cannot import 'aeon'. Install with: pip install aeon"
    ) from e

# --- SciPy --------------------------------------------------------------------
from scipy.stats import (
    friedmanchisquare,
    f,
    wilcoxon,
    norm,
)

# =============================================================================
# I/O & PREP
# =============================================================================
def _ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _infer_model_name(path: str) -> str:
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    return name


def _to_numeric_clean(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )
    return pd.to_numeric(s, errors="coerce")

def load_model_csvs(
    folder: str,
    required_columns: tuple[str, ...] = ("AUC (%)", "F1 Score (%)", "Time Training (s)"),
    with_scorecard: bool = False
) -> pd.DataFrame:
    records = []
    for fn in os.listdir(folder):
        if not fn.lower().endswith(".csv"):
            continue
        if not with_scorecard and "rulecard" in fn.lower():
            continue
        path = os.path.join(folder, fn)
        model = _infer_model_name(path)
        df = pd.read_csv(path, index_col=0)

        if df.index.name is None or str(df.index.name).startswith("Unnamed"):
            df.index.name = "dataset_name"
        df["dataset_name"] = df.index.astype(str)
        df["classifier_name"] = model.replace("_comparison", "").upper()
        records.append(df.reset_index(drop=True))

    if not records:
        raise ValueError(f"Nessun CSV trovato in {folder}")

    df_all = pd.concat(records, ignore_index=True)

    missing = [c for c in required_columns if c not in df_all.columns]
    if missing:
        raise ValueError(f"Colonne richieste mancanti nei CSV: {missing}")

    keep_cols = ["dataset_name", "classifier_name"] + list(required_columns)
    df_all = df_all[keep_cols].copy()

    for c in required_columns:
        df_all[c] = _to_numeric_clean(df_all[c])

    return df_all

def _prepare_scores(
    df_all: pd.DataFrame, metric: str
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Datasets with NaN are removed to respect the complete block design
    (Demšar 2006). In case of incomplete blocks, classical Friedman is not applicable.
    """
    if metric not in df_all.columns:
        raise ValueError(
            f"Metric '{metric}' not found. Columns: {list(df_all.columns)}"
        )

    pivot = (
        df_all.pivot_table(
            index="dataset_name", columns="classifier_name", values=metric, aggfunc="mean"
        )
        .sort_index(axis=1)
        .copy()
    )

    initial_n = pivot.shape[0]
    pivot = pivot.dropna(axis=0, how="any")
    dropped = initial_n - pivot.shape[0]
    if dropped > 0:
        print(f"[{metric}] Removed {dropped} datasets with missing values (complete blocks required).")

    if pivot.shape[0] < 2:
        raise ValueError(f"[{metric}] At least 2 datasets are required (found {pivot.shape[0]}).")
    if pivot.shape[1] < 2:
        raise ValueError(f"[{metric}] At least 2 classifiers are required (found {pivot.shape[1]}).")

    scores = pivot.to_numpy()
    labels = list(pivot.columns)
    print(f"[{metric}] datasets={scores.shape[0]} | classifiers={scores.shape[1]}")
    return scores, labels, pivot


# =============================================================================
# STAT: OMNIBUS + POST-HOC
# =============================================================================
def friedman_report(pivot: pd.DataFrame) -> dict:
    """
    Friedman omnibus + Iman-Davenport correction
    If k=2, use Wilcoxon for paired samples.

    - Iman & Davenport (1980): F-approx
        F_F = ((N-1) * chi2_F) / (N*(k-1) - chi2_F)
      with df1=k-1, df2=(k-1)*(N-1)
    """
    cols = list(pivot.columns)
    k, N = len(cols), len(pivot)

    if k == 2:
        a, b = pivot[cols[0]].values, pivot[cols[1]].values
        try:
            w_stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            w_stat, p = 0.0, 1.0
        print(
            f"[Wilcoxon] {cols[0]} vs {cols[1]}: W={w_stat:.3f}, p={p:.4g} (N={N})"
        )
        return {"test": "wilcoxon", "p": float(p), "k": k, "N": N}

    chi2, p_chi2 = friedmanchisquare(*[pivot[c].values for c in cols])
    Ff = ((N - 1) * chi2) / (N * (k - 1) - chi2)  # Iman-Davenport
    p_F = f.sf(Ff, k - 1, (k - 1) * (N - 1))
    print(
        f"[Friedman] chi2={chi2:.3f}, p={p_chi2:.4g} | "
        f"[Iman-Davenport] F={Ff:.3f}, p={p_F:.4g} (k={k}, N={N})"
    )
    return {
        "test": "friedman",
        "chi2": float(chi2),
        "p_chi2": float(p_chi2),
        "F_ID": float(Ff),
        "p_ID": float(p_F),
        "k": k,
        "N": N,
    }


def _holm_adjust(pvals: list[float]) -> list[float]:
    """
    Holm step-down correction (Holm, 1979).
    Monotonic and more powerful than single Bonferroni. Maintains FWER.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    prev = 0.0
    for i, idx in enumerate(order):
        adj = (m - i) * pvals[idx]
        adj = max(adj, prev)
        adjusted[idx] = min(adj, 1.0)
        prev = adjusted[idx]
    return adjusted.tolist()


def posthoc_vs_control(
    pivot: pd.DataFrame,
    control: str,
    lower_better: bool,
    alpha: float = 0.05,
    correction: str = "holm",
) -> pd.DataFrame:
    """
    Post-hoc vs control, based on differences in average ranks (Demšar 2006):
      z = |R_i - R_j| / sqrt( k(k+1) / (6N) )
    two-sided p-value from N(0,1). p adjusted with Holm (default).

    Dunn test vs control in terms of ranks (Demšar 2006).
    """
    assert control in pivot.columns, f"Control '{control}' not found in {list(pivot.columns)}"

    ranks = pivot.rank(axis=1, method="average", ascending=lower_better)
    avg_ranks = ranks.mean(axis=0)

    cols = list(pivot.columns)
    k, N = len(cols), len(pivot)
    se = np.sqrt(k * (k + 1) / (6.0 * N))

    rows = []
    for clf in cols:
        if clf == control:
            continue
        diff = abs(avg_ranks[clf] - avg_ranks[control])
        z = diff / se
        p = 2 * norm.sf(z)
        rows.append(
            {
                "classifier": clf,
                "avg_rank": float(avg_ranks[clf]),
                "diff_from_control": float(diff),
                "z": float(z),
                "p_raw": float(p),
            }
        )

    df = pd.DataFrame(rows).sort_values("p_raw", ascending=True)
    if correction and correction.lower() == "holm" and len(df) > 0:
        df["p_holm"] = _holm_adjust(df["p_raw"].tolist())
        df["significant"] = df["p_holm"] < alpha
    else:
        df["p_holm"] = np.nan
        df["significant"] = df["p_raw"] < alpha

    return df[
        ["classifier", "avg_rank", "diff_from_control", "z", "p_raw", "p_holm", "significant"]
    ]


# =============================================================================
# CD-DIAGRAM (Nemenyi or Dunn)
# =============================================================================
def make_cd_plot(
    df_all: pd.DataFrame,
    metric: str,
    lower_better: bool,
    alpha: float = 0.05,
    out_prefix: str | None = None,
    reverse: bool = True,
    force_test: str | None = None,
    png_dpi: int = 300,
    save_pdf: bool = False,
    control: str | None = None,
    export_posthoc_table: bool = True,
    outdir: str = "con_scorecard",
) -> None:
    """
    Create and save the CD diagram for the specified metric.

    - The CD plot implements Nemenyi (all-vs-all) or Dunn/Bonferroni-Dunn (vs control).
    - Post-hoc tables can use Holm (more powerful), but this is NOT visualizable with
      a single CD line; expect divergences between plot and table (to be explained in the text).
    """
    _ensure_outdir(outdir)
    scores, labels, pivot = _prepare_scores(df_all, metric)

    omnibus = friedman_report(pivot)
    k = omnibus["k"]

    if out_prefix is None:
        safe_metric = (
            metric.lower()
            .replace("%", "perc")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
            .replace("/", "_")
        )
        out_prefix = f"cd_{safe_metric}"

    if control is not None:
        assert control in labels, f"Controllo '{control}' non tra {labels}"
        test_to_use = "wilcoxon" if force_test is None else force_test
    else:
        test_to_use = "nemenyi" if force_test is None else force_test

    if control is not None and k >= 3 and export_posthoc_table:
        post = posthoc_vs_control(
            pivot, control=control, lower_better=lower_better, alpha=alpha, correction="holm"
        )
        csv_path = os.path.join(outdir, f"{out_prefix}_posthoc_vs_{control}.csv")
        post.to_csv(csv_path, index=False)
        print(f"[{metric}] Post-hoc vs '{control}' salvato in: {csv_path}")

    kwargs = dict(
        lower_better=lower_better,
        alpha=alpha,
        test=test_to_use,
        reverse=reverse,
        return_p_values=False,
    )

    fig_ax = plot_critical_difference(scores, labels, kwargs)
    fig, ax = fig_ax[:2]

    title_suffix = ""
    if omnibus["test"] == "friedman":
        p_id = omnibus["p_ID"]
        title_suffix = f" — Friedman/Iman-Davenport p={p_id:.3g}"
    elif omnibus["test"] == "wilcoxon":
        title_suffix = f" — Wilcoxon p={omnibus['p']:.3g}"
    ax.set_title(f"CD Diagram - {metric}{title_suffix}", fontsize=14)

    png_path = os.path.join(outdir, f"{out_prefix}.png")
    fig.savefig(png_path, bbox_inches="tight", dpi=png_dpi)
    if save_pdf:
        pdf_path = os.path.join(outdir, f"{out_prefix}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[{metric}] CD salvato: {png_path}" + (" (+ PDF)" if save_pdf else ""))


def make_mcm_plot(
    df_all: pd.DataFrame,
    metric: str,
    lower_better: bool,
    out_prefix: str | None = None,
    save_formats: tuple[str, ...] = ("png",),
    alpha: float = 0.05,
    outdir: str = "con_scorecard",
) -> None:
    """
    Generate the Multiple Comparison Matrix (MCM) for the specified metric.

    Here we use Wilcoxon for pairs with Holm correction (more powerful than Nemenyi),
    useful as a “ground truth” for decisions, leaving the CD plot as a visual
    descriptive of the ranks (García & Herrera, 2008).
    """
    _ensure_outdir(outdir)
    _, _, pivot = _prepare_scores(df_all, metric)

    if out_prefix is None:
        safe_metric = (
            metric.lower()
            .replace("%", "perc")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
            .replace("/", "_")
        )
        out_prefix = f"mcm_{safe_metric}"

    df_for_mcm = pivot.reset_index()
    fig = create_multi_comparison_matrix(
        df_for_mcm,
        save_path=os.path.join(outdir, out_prefix),
        formats=list(save_formats),
        used_statistic=metric,
        higher_stat_better=not lower_better,
        include_pvalue=True,
        pvalue_test="wilcoxon",
        pvalue_correction="holm",
        pvalue_threshold=alpha,
        dataset_column="dataset_name",
        precision=4,
        colormap="coolwarm",
        fig_size=(22, 8),
        font_size=16,
        include_legend=True,
        show_symetry=True,
    )
    plt.close(fig)
    print(f"[{metric}] MCM saved as {save_formats} in {os.path.join(outdir, out_prefix)}.*")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    FOLDER = "RESULTS"
    with_scorecard = False
    OUTDIR = "csv_files_img"

    BASELINE = None
    suffix = ""
    if with_scorecard:
        BASELINE = "RULECARD"
        suffix = "_rulecard"

    METRICS = [
        ("AUC (%)", False),
        ("F1 Score (%)", False),
        ("Time Training (s)", True),
    ]

    df_all = load_model_csvs(
        FOLDER,
        required_columns=tuple(m for m, _ in METRICS),
        with_scorecard=with_scorecard,
    )

    # CD-diagrams
    for metric, lower_better in METRICS:
        make_cd_plot(
            df_all,
            metric=metric,
            lower_better=lower_better,
            alpha=0.05,
            reverse=True,
            force_test=None,
            png_dpi=300,
            save_pdf=False,
            control=BASELINE,
            export_posthoc_table=True,
            outdir=OUTDIR,
            out_prefix=f"CD_{metric.replace(' ', '_').replace('%','perc')}{suffix}",
        )

    # MCM plots
    for metric, lower_better in METRICS:
        make_mcm_plot(
            df_all,
            metric=metric,
            lower_better=lower_better,
            save_formats=("png",),
            alpha=0.05,
            outdir=OUTDIR,
            out_prefix=f"MCM_{metric.replace(' ', '_').replace('%','perc')}{suffix}",
        )
