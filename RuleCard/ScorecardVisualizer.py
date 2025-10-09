from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from matplotlib.text import Text



import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from collections import defaultdict, OrderedDict
import re, textwrap
from matplotlib.table import Cell


import matplotlib.pyplot as plt
from matplotlib import colors, colormaps


@dataclass
class ScorecardVisualizer:
    scorecard: Any
    scaler: Any = None
    scaler_feature_names: Optional[List[str]] = None
    mapping: Optional[Dict[str, List[str]]] = None
    sig_digits: int = 2
    
    _ohe_lookup: Dict[str, Tuple[str, str]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.mapping:
            for fam, cols in self.mapping.items():
                prefix = f"{fam}_"
                for col in cols:
                    base = self._strip_prefix(str(col))
                    cat = base[len(prefix):] if base.startswith(prefix) else base.replace(prefix, "", 1)
                    self._ohe_lookup[base] = (fam, cat)

    def _split_ohe(self, name: str) -> Optional[Tuple[str, str]]:
        base = self._strip_prefix(str(name))
        return self._ohe_lookup.get(base)

    def _score_from_proba(self, p: float) -> float:
        sc = self.scorecard
        return float(sc.offset) + float(sc.factor) * np.log(float(p)/(1.0-float(p)))

    def compute_risk_bands(self, *, probs=None, labels=None, colors=None) -> Dict[str, Any]:
        if probs is None:
            probs = [0.05, 0.15, 0.35, 0.65]
        probs = sorted(float(p) for p in probs if 0.0 < p < 1.0)

        cut_scores = [self._score_from_proba(p) for p in probs]

        nbands = len(probs) + 1
        if labels is None:
            labels = ["Very Low","Low","Medium","High","Very High"][:nbands]
        if colors is None:
            colors = (["#d73027","#fc8d59","#fee08b","#91cf60","#1a9850"]
                    if nbands==5 else ["#d73027","#fdae61","#a6d96a","#1a9850"][:nbands])

        s_min = self._score_from_proba(0.01)
        s_max = self._score_from_proba(0.99)

        return {
            "edges_proba": [0.0] + probs + [1.0],
            "edges_score": [None] + cut_scores + [None],
            "labels": labels,
            "colors": colors,
            "threshold_score": float(self.scorecard.offset),
            "score_min": float(s_min),
            "score_max": float(s_max),
        }




    def _families_from_features(self, feats: Any) -> List[str]:
        fams = []
        if isinstance(feats, (list, tuple)):
            for f in feats:
                base = self._strip_prefix(str(f))
                fc = self._split_ohe(base)
                fam = fc[0] if fc else base
                fams.append(self._clean_identifier(fam))
        fams = sorted(list({f for f in fams if f}))
        return fams[:2] if fams else ["Other"]

    def compute_factors_share(
        self, *, top_k: int = 5
    ) -> Dict[str, Any]:
        df = self.scorecard.rules_df_points.copy()
        if "support" not in df.columns or "points" not in df.columns:
            return {"items": [], "total_abs_impact": 0.0}

        def _label_from_row(row) -> str:
            fams = self._families_from_features(row.get("features", None))
            return " & ".join(fams) if len(fams) > 1 else fams[0]

        df["__family__"] = df.apply(_label_from_row, axis=1)
        df["__impact__"] = (df["support"].astype(float) * df["points"].astype(float)).abs()

        byfam = (df.groupby("__family__", as_index=False)
                .agg(abs_impact=("__impact__", "sum"),
                        n_rules=("points", "count")))

        total = float(byfam["abs_impact"].sum()) or 1.0
        byfam["share"] = byfam["abs_impact"] / total
        byfam = byfam.sort_values("share", ascending=False)

        top = byfam.head(top_k).to_dict("records")
        if len(byfam) > top_k:
            other_share = float(byfam["share"].iloc[top_k:].sum())
            other_impact = float(byfam["abs_impact"].iloc[top_k:].sum())
            top.append({"__family__": "Other", "abs_impact": other_impact, "n_rules": int(byfam["n_rules"].iloc[top_k:].sum()), "share": other_share})

        items = [{"name": r["__family__"], "share": float(r["share"]), "abs_impact": float(r["abs_impact"]), "n_rules": int(r["n_rules"])} for r in top]
        return {"items": items, "total_abs_impact": float(total)}


    @staticmethod
    def _expit(x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

    @staticmethod
    def _predict_proba_from_score(S, *, factor, offset):
        z = (np.asarray(S, dtype=float) - float(offset)) / float(factor)
        p1 = ScorecardVisualizer._expit(z)
        return np.column_stack([1.0 - p1, p1])

    def _layout_profile(self, which: Literal["global", "local"]) -> Dict[str, Any]:
        if which == "global":
            return dict(
                wrap_width=70,
                base_row_h=0.06,
                row_gap=0.015,
                min_row_inch=0.02,
                cell_pad=0.01,
                col_widths=(0.9, 0.1),
                font_min=8,
                font_max=11,
                max_fig_height=10,
            )

        return dict(
            wrap_width=58,
            base_row_h=0.1,
            row_gap=0.015,
            min_row_inch=0.03, 
            cell_pad=0.01,
            col_widths=(0.9, 0.1),
            font_min=9,
            font_max=12,
            max_fig_height=10,
        )
    
    

    def _inverse_feature_values(self, var: str, values: List[float]) -> List[float]:
        if self.scaler is None:
            return values
        s = self.scaler
        n = getattr(s, "n_features_in_", None)

        if hasattr(s, "feature_names_in_"):
            names = list(s.feature_names_in_)
        elif self.scaler_feature_names is not None and (n is None or len(self.scaler_feature_names) == n):
            names = list(self.scaler_feature_names)
        else:
            return values
        if n is None:
            n = len(names)

        candidates = [var, self._strip_prefix(var)]
        idx = next((i for i, c in enumerate(names) if c in candidates), None)
        if idx is None:
            return values

        if hasattr(s, "mean_") and hasattr(s, "scale_"):
            mean = float(s.mean_[idx])
            scale = float(s.scale_[idx]) if getattr(s, "with_std", True) else 1.0
            return [mean + float(v) * scale for v in values]

        if hasattr(s, "data_min_") and hasattr(s, "data_range_") and hasattr(s, "feature_range"):  # MinMaxScaler
            data_min = float(s.data_min_[idx])
            data_rng = float(s.data_range_[idx])
            lo, hi = s.feature_range
            rng = float(hi - lo) if hi != lo else 1.0
            return [data_min + (float(v) - lo) / rng * data_rng for v in values]

        try:
            arr = np.zeros((1, n), dtype=float)
            out = []
            for v in values:
                arr.fill(0.0)
                arr[0, idx] = float(v)
                inv = s.inverse_transform(arr)
                out.append(float(inv[0, idx]))
            return out
        except Exception:
            return values


    def _inverse_thresholds(self, expr: str) -> str:
        if self.scaler is None:
            return expr

        num = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        var = r"[A-Za-z0-9_.]+"
        ws  = r"\s*"

        # 1) interval: num < var <= num
        pat_interval = re.compile(
            rf"(?P<lo>{num}){ws}(?P<op1><=|<){ws}(?P<var>{var}){ws}(?P<op2><=|<){ws}(?P<hi>{num})",
            re.IGNORECASE
        )
        def repl_interval(m):
            lo = float(m.group("lo")); hi = float(m.group("hi")); vr = m.group("var")
            lo_i, hi_i = self._inverse_feature_values(vr, [lo, hi])
            return f"{self._fmt_num(lo_i[0])} {m.group('op1')} {vr} {m.group('op2')} {self._fmt_num(hi_i[1])}"
        s = pat_interval.sub(repl_interval, str(expr))

        # 2) var <op> num
        pat_left = re.compile(rf"(?P<var>{var}){ws}(?P<op><=|>=|<|>){ws}(?P<v>{num})", re.IGNORECASE)
        def repl_left(m):
            vr, op, v = m.group("var"), m.group("op"), float(m.group("v"))
            inv = self._inverse_feature_values(vr, [v])[0]
            return f"{vr} {op} {self._fmt_num(inv)}"
        s = pat_left.sub(repl_left, s)

        # 3) num <op> var
        pat_right = re.compile(rf"(?P<v>{num}){ws}(?P<op><=|>=|<|>){ws}(?P<var>{var})", re.IGNORECASE)
        def repl_right(m):
            v, op, vr = float(m.group("v")), m.group("op"), m.group("var")
            inv = self._inverse_feature_values(vr, [v])[0]
            return f"{self._fmt_num(inv)} {op} {vr}"
        s = pat_right.sub(repl_right, s)

        return s
    
    _num_pat_interval = re.compile(
        r"^\s*(?P<lo>[-+]?\d+(?:\.\d+)?)\s*(?P<op1><=|<)\s*(?P<var>[A-Za-z0-9_.]+)\s*(?P<op2><=|<)\s*(?P<hi>[-+]?\d+(?:\.\d+)?)\s*$"
    )
    _num_pat_left = re.compile(
        r"^\s*(?P<var>[A-Za-z0-9_.]+)\s*(?P<op><=|>=|<|>)\s*(?P<v>[-+]?\d+(?:\.\d+)?)\s*$"
    )
    _num_pat_right = re.compile(
        r"^\s*(?P<v>[-+]?\d+(?:\.\d+)?)\s*(?P<op><=|>=|<|>)\s*(?P<var>[A-Za-z0-9_.]+)\s*$"
    )

    _num_pat = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

    def _shrink_numbers_in_str(self, s: str) -> str:
        return self._num_pat.sub(lambda m: self._fmt_num(float(m.group(0))), str(s))

    @staticmethod
    def _merge_intervals(bins: List[Tuple[Optional[float], bool, Optional[float], bool, float]]
                        ) -> List[Tuple[Optional[float], bool, Optional[float], bool, float]]:
        if not bins:
            return []

        norm_bins = []
        for lo, lo_inc, hi, hi_inc, pts in bins:
            lo_val = -float("inf") if lo is None else lo
            hi_val = float("inf") if hi is None else hi
            norm_bins.append((lo_val, lo_inc, hi_val, hi_inc, pts))

        norm_bins.sort(key=lambda x: (x[0], not x[1]))

        merged = []
        for lo, lo_inc, hi, hi_inc, pts in norm_bins:
            if not merged:
                merged.append([lo, lo_inc, hi, hi_inc, pts])
                continue

            mlo, mlo_inc, mhi, mhi_inc, mpts = merged[-1]

            if lo < mhi or (lo == mhi and (lo_inc or mhi_inc)):
                new_hi = max(mhi, hi)
                if hi > mhi:
                    new_hi_inc = hi_inc
                elif hi == mhi:
                    new_hi_inc = hi_inc or mhi_inc
                else:
                    new_hi_inc = mhi_inc
                merged[-1] = [mlo, mlo_inc, new_hi, new_hi_inc, mpts + pts]
            else:
                merged.append([lo, lo_inc, hi, hi_inc, pts])

        out = []
        for lo, lo_inc, hi, hi_inc, pts in merged:
            lo_val = None if lo == -float("inf") else lo
            hi_val = None if hi == float("inf") else hi
            out.append((lo_val, lo_inc, hi_val, hi_inc, pts))

        collapsed = {}
        for lo, lo_inc, hi, hi_inc, pts in out:
            key = (lo, lo_inc, hi, hi_inc)
            if key not in collapsed:
                collapsed[key] = pts
            else:
                collapsed[key] += pts

        out_final = [(lo, lo_inc, hi, hi_inc, pts) for (lo, lo_inc, hi, hi_inc), pts in collapsed.items()]
        return out_final



    def _format_interval_label(self, var: str, lo, lo_inc, hi, hi_inc) -> str:
        v = self._clean_identifier(var)
        if lo is None and hi is None:
            return v
        if lo is None:
            br = "≤" if hi_inc else "<"
            return f"{v} {br} {self._fmt_num(hi)}"
        if hi is None:
            br = "≥" if lo_inc else ">"
            return f"{v} {br} {self._fmt_num(lo)}"

        left = "≤" if lo_inc else "<"
        right = "≤" if hi_inc else "<"
        return f"{self._fmt_num(lo)} {left} {v} {right} {self._fmt_num(hi)}"

    def _fmt_num(self, v: float) -> str:
        v = float(v)
        if np.isfinite(v) and abs(v - round(v)) < 1e-9:
            return str(int(round(v)))

        if v == 0.0:
            return "0"

        sig = max(1, int(self.sig_digits))
        dec = max(0, sig - 1 - int(np.floor(np.log10(abs(v)))))

        s = f"{v:.{dec}f}"
        s = s.rstrip("0").rstrip(".")
        return s


    
    def global_table(
        self,
        *,
        columns: Optional[List[str]] = None,
        sort_by: Union[str, Sequence[str]] = ("stage", "points"),
        ascending: Union[bool, Sequence[bool]] = (True, False),
    ) -> pd.DataFrame:

        df = self.scorecard.rules_df_points.copy()

        default_cols = ["stage", "kind", "expr", "points", "weight",
                    "support", "pos_rate", "features", "bounds"]
        cols = [c for c in (columns or default_cols) if c in df.columns]
        if not cols:
            cols = [c for c in default_cols if c in df.columns]

        out = df[cols].copy()

        if "expr" in out.columns:
            out["expr"] = out["expr"].map(lambda e: self._shrink_numbers_in_str(self._inverse_thresholds(str(e))))

        if sort_by is not None:
            if isinstance(sort_by, str):
                sort_by = [sort_by]
            if isinstance(ascending, bool):
                ascending = [ascending] * len(sort_by)
            sort_by = [c for c in sort_by if c in out.columns]
            if sort_by:
                out = out.sort_values(by=sort_by, ascending=ascending, kind="mergesort")

        return out.reset_index(drop=True)

    def local_table(
        self,
        x: Union[pd.Series, pd.DataFrame, Dict[str, Any]],
        *,
        bounds_col: str = "bounds",
        on_missing: str = "error",
        return_sparse: bool = True,
    ) -> Tuple[pd.DataFrame, int, np.ndarray, int]:

        sc = self.scorecard

        if isinstance(x, dict):
            x = pd.DataFrame([x])
        elif isinstance(x, pd.Series):
            x = x.to_frame().T
        elif isinstance(x, pd.DataFrame):
            if x.shape[0] != 1:
                raise ValueError("In local_table, 'x' deve avere esattamente 1 riga.")
        else:
            raise ValueError("x dev'essere dict, Series o DataFrame.")

        A_one = sc.make_activation_matrix(x, sc.rules_df_points, dtype=np.int8)
        active_mask = (A_one.toarray().ravel() > 0) if hasattr(A_one, "toarray") else (np.asarray(A_one).ravel() > 0)

        df_act = sc.rules_df_points.loc[active_mask].copy().reset_index(drop=True)
        df_act = df_act.sort_values(by=["stage", "points"], ascending=[True, False], kind="mergesort")

        S_points = int(sc.base_points) + int(df_act["points"].sum())

        base_row = {
            "stage": -1, "kind": "[BASE]", "expr": "[BASE]",
            "points": int(sc.base_points), "features": None, "bounds": None
        }
        cols_show = ["stage", "kind", "expr", "points", "features", "bounds"]
        for extra in ["weight", "support", "pos_rate"]:
            if extra in df_act.columns and extra not in cols_show:
                cols_show.append(extra)

        df_local = pd.concat([pd.DataFrame([base_row]), df_act[cols_show]], ignore_index=True)

        proba = self._predict_proba_from_score([S_points], factor=sc.factor, offset=sc.offset)[0]
        yhat = int(proba[1] >= 0.5)

        if "expr" in df_local.columns:
            df_local["expr"] = df_local["expr"].map(
                lambda e: self._shrink_numbers_in_str(self._inverse_thresholds(str(e)))
            )
        return df_local, S_points, proba, yhat

    @staticmethod
    def _ensure_dataframe(df_in: Union[pd.DataFrame, Sequence[Dict[str, Any]]]) -> pd.DataFrame:
        if isinstance(df_in, pd.DataFrame):
            return df_in.copy()
        if isinstance(df_in, (list, tuple)):
            return pd.DataFrame(list(df_in))
        raise ValueError("df_in dev'essere DataFrame o Sequence[dict].")

    @staticmethod
    def _strip_prefix(name: str) -> str:
        if name.startswith("cat_"):
            return name[len("cat_"):]
        if name.startswith("num_"):
            return name[len("num_"):]
        return name
    
    @staticmethod
    def _clean_identifier(name: str) -> str:
        s = ScorecardVisualizer._strip_prefix(str(name))
        s = re.sub(r"_+", " ", s).strip()
        return s

    def _clean_expr_for_display(self, expr: str) -> str:
        token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
        def repl(m):
            return self._clean_identifier(m.group(0))
        return token_re.sub(repl, str(expr))

    def _other_label(
        self,
        fam: str,
        cat: str,
        cats_known: List[str]
    ) -> Optional[Union[str, List[str]]]:
        if not cats_known:
            return None

        others = [c for c in cats_known if c != cat]

        if not others:
            return None
        if len(cats_known) == 2:
            return others[0]
        return others


    def _collect_onehot_vocab(self, df_any: pd.DataFrame) -> Dict[str, List[str]]:
        cats = defaultdict(set)

        if "features" in df_any.columns:
            for feats in df_any["features"]:
                if isinstance(feats, (list, tuple)):
                    for f in feats:
                        base = self._strip_prefix(str(f))
                        fc = self._split_ohe(base)
                        if fc:
                            fam, cat = fc
                            cats[fam].add(cat)

        if "condition" in df_any.columns:
            pat = re.compile(r"([A-Za-z][A-Za-z0-9_]+)\s*(?:==|!=|:)\s*(?:0|1|YES|NO|True|False)", re.IGNORECASE)
            for cond in df_any["condition"].astype(str):
                for m in pat.finditer(cond):
                    tok = self._strip_prefix(m.group(1))
                    fc = self._split_ohe(tok)
                    if fc:
                        fam, cat = fc
                        cats[fam].add(cat)

        return {k: sorted(v) for k, v in cats.items()}


    @staticmethod
    def _is_true_token(tok: str) -> bool:
        return tok.lower() in ("1", "yes", "true")

    def _cond_pretty(self, cond: str, fam_label: str, vocab: Dict[str, List[str]]) -> str:
        cats_obs = vocab.get(fam_label, [])
        if self.mapping and fam_label in self.mapping:
            cats_map = [self._strip_prefix(c).replace(f"{fam_label}_", "", 1) for c in self.mapping[fam_label]]
            binary_like = (len(set(cats_map)) == 2)
        else:
            binary_like = (len(cats_obs) == 2)

        val_re = r"(?:0|1|YES|NO|True|False)"
        m = re.fullmatch(
            rf"{re.escape(fam_label)}_([A-Za-z0-9_]+)\s*(==|!=|:)\s*({val_re})",
            cond.strip(), flags=re.IGNORECASE
        )
        if not m:
            return cond

        cat, op, tok = m.group(1), m.group(2), m.group(3)
        op = "==" if op == ":" else op
        present = self._is_true_token(tok)
        other = self._other_label(fam_label, cat, cats_obs)

        if op == "==":
            if present:
                return f"{fam_label} {cat}" if binary_like else f"{fam_label} = {cat}"
            return (f"{fam_label} {other}" if (binary_like and other is not None)
                    else f"{fam_label} ≠ {cat}")
        else:
            if present:
                return (f"{fam_label} {other}" if (binary_like and other is not None)
                        else f"{fam_label} ≠ {cat}")
            return f"{fam_label} {cat}" if binary_like else f"{fam_label} = {cat}"

    def _parse_numeric_interval(
        self, s: str
    ) -> Optional[Tuple[str, Optional[float], bool, Optional[float], bool]]:
        t = str(s).strip()

        # num < var <= num
        m = self._num_pat_interval.fullmatch(t)
        if m:
            lo = float(m.group("lo")); hi = float(m.group("hi"))
            var = m.group("var")
            lo_inc = (m.group("op1") == "<=")
            hi_inc = (m.group("op2") == "<=")
            return var, lo, lo_inc, hi, hi_inc

        # var <op> num
        m = self._num_pat_left.fullmatch(t)
        if m:
            var, op, v = m.group("var"), m.group("op"), float(m.group("v"))
            if op == "<":   return var, None, True,  v, False
            if op == "<=":  return var, None, True,  v, True
            if op == ">":   return var, v,    False, None, True
            if op == ">=":  return var, v,    True,  None, True

        # num <op> var
        m = self._num_pat_right.fullmatch(t)
        if m:
            v, op, var = float(m.group("v")), m.group("op"), m.group("var")
            if op == "<":   return var, v, False, None, True         # v < var  -> lo=(v, open)
            if op == "<=":  return var, v, True,  None, True         # v <= var -> lo=(v, closed)
            if op == ">":   return var, None, True, v, False         # v > var  -> var < v
            if op == ">=":  return var, None, True, v, True          # v >= var -> var <= v

        return None


    def _cond_pretty_all(self, cond: str, vocab: Dict[str, List[str]]) -> str:
        pat_exact = re.compile(
            r"(?P<tok>[A-Za-z][A-Za-z0-9_]+)"
            r"\s*(?P<op>:|==|!=)\s*"
            r"(?P<val>0|1|YES|NO|True|False)",
            re.IGNORECASE,
        )
        pat_threshold = re.compile(
            r"(?P<tok>[A-Za-z][A-Za-z0-9_]+)"
            r"\s*(?P<op><=|>=|<|>)\s*"
            r"(?P<val>(?:0|1)(?:\.\d+)?)", 
            re.IGNORECASE,
        )
        pat_space_exact = re.compile(
            r"(?P<fam>[A-Za-z][A-Za-z0-9_]*)\s+"
            r"(?P<cat>[A-Za-z0-9_]+)\s*"
            r"(?P<op>:|==|!=)\s*"
            r"(?P<val>0|1|YES|NO|True|False)",
            re.IGNORECASE,
        )
        pat_space_threshold = re.compile(
            r"(?P<fam>[A-Za-z][A-Za-z0-9_]*)\s+"
            r"(?P<cat>[A-Za-z0-9_]+)\s*"
            r"(?P<op><=|>=|<|>)\s*"
            r"(?P<val>(?:0|1)(?:\.\d+)?)",
            re.IGNORECASE,
        )

        def _render_yesno(fam: str, cat: str, present: bool) -> str:
            return f"{self._clean_identifier(fam)} {self._clean_identifier(cat)} {'YES' if present else 'NO'}"

        def repl_exact(m):
            tok = m.group("tok")
            op  = "==" if m.group("op") == ":" else m.group("op")
            val = m.group("val")
            base = self._strip_prefix(tok)
            fc = self._split_ohe(base)
            if not fc:
                return m.group(0)
            fam, cat = fc
            present = self._is_true_token(val)
            if op == "!=":
                present = not present
            return _render_yesno(fam, cat, present)

        def repl_threshold(m):
            tok = m.group("tok")
            op  = m.group("op")
            thr = float(m.group("val"))
            base = self._strip_prefix(tok)
            fc = self._split_ohe(base)
            if not fc:
                return m.group(0)
            fam, cat = fc
            eps = 1e-9
            if op in (">", ">=") and thr >= 0.5 - eps:
                return _render_yesno(fam, cat, True)
            if op in ("<", "<=") and thr <= 0.5 + eps:
                return _render_yesno(fam, cat, False)
            return m.group(0)

        def repl_space_exact(m):
            fam = m.group("fam"); cat = m.group("cat")
            op  = "==" if m.group("op") == ":" else m.group("op")
            val = m.group("val")
            present = self._is_true_token(val)
            if op == "!=":
                present = not present
            return _render_yesno(fam, cat, present)

        def repl_space_threshold(m):
            fam = m.group("fam"); cat = m.group("cat")
            op  = m.group("op")
            thr = float(m.group("val"))
            eps = 1e-9
            if op in (">", ">=") and thr >= 0.5 - eps:
                return _render_yesno(fam, cat, True)
            if op in ("<", "<=") and thr <= 0.5 + eps:
                return _render_yesno(fam, cat, False)
            return m.group(0)

        result = cond
        result = pat_exact.sub(repl_exact, result)
        result = pat_threshold.sub(repl_threshold, result)
        result = pat_space_exact.sub(repl_space_exact, result)
        result = pat_space_threshold.sub(repl_space_threshold, result)
        return result

    @staticmethod
    def _merge_active_bins(bins):
        if not bins:
            return []

        norm_bins = []
        for lo, lo_inc, hi, hi_inc, pts in bins:
            lo_val = -float("inf") if lo is None else lo
            hi_val = float("inf") if hi is None else hi
            norm_bins.append((lo_val, lo_inc, hi_val, hi_inc, pts))

        norm_bins.sort(key=lambda x: (x[0], not x[1]))

        merged = []
        for lo, lo_inc, hi, hi_inc, pts in norm_bins:
            if not merged:
                merged.append([lo, lo_inc, hi, hi_inc, pts])
                continue
            mlo, mlo_inc, mhi, mhi_inc, mpts = merged[-1]
            if lo < mhi or (lo == mhi and (lo_inc or mhi_inc)):
                new_hi = max(mhi, hi)
                if hi > mhi:
                    new_hi_inc = hi_inc
                elif hi == mhi:
                    new_hi_inc = hi_inc or mhi_inc
                else:
                    new_hi_inc = mhi_inc
                merged[-1] = [mlo, mlo_inc, new_hi, new_hi_inc, mpts + pts]
            else:
                merged.append([lo, lo_inc, hi, hi_inc, pts])

        out = []
        for lo, lo_inc, hi, hi_inc, pts in merged:
            lo_val = None if lo == -float("inf") else lo
            hi_val = None if hi == float("inf") else hi
            out.append((lo_val, lo_inc, hi_val, hi_inc, pts))
        return out

    @staticmethod
    def _partition_and_merge_intervals(
        bins: List[Tuple[Optional[float], bool, Optional[float], bool, float]]
    ) -> List[Tuple[Optional[float], bool, Optional[float], bool, float]]:

        if not bins:
            return []

        cuts = set()
        for lo, _, hi, _, _ in bins:
            if lo is not None and np.isfinite(lo):
                cuts.add(lo)
            if hi is not None and np.isfinite(hi):
                cuts.add(hi)
        cuts = sorted(cuts)

        bounds = [-float("inf")] + cuts + [float("inf")]

        out = []
        for i in range(len(bounds) - 1):
            lo, hi = bounds[i], bounds[i + 1]

            #lo_inc = (lo == -float("inf"))
            #hi_inc = (hi == float("inf"))
            lo_inc = True
            hi_inc = True

            pts_tot = 0.0
            for blo, blo_inc, bhi, bhi_inc, bpts in bins:
                lo_ok = (blo is None or blo < hi or (blo == hi and blo_inc and hi_inc))
                hi_ok = (bhi is None or bhi > lo or (bhi == lo and bhi_inc and lo_inc))
                if lo_ok and hi_ok:
                    pts_tot += bpts

            out.append((
                None if lo == -float("inf") else lo,
                lo_inc,
                None if hi == float("inf") else hi,
                hi_inc,
                pts_tot
            ))

        out = [iv for iv in out if abs(iv[4]) > 1e-12]
        return out


    def _build_scorecard_table_data(
        self,
        df_in: Union[pd.DataFrame, Sequence[Dict[str, Any]]],
        *,
        include_pairs: bool = True,
        sort_bins_by: str = "points_desc",
        selected_rule_ids: Optional[Iterable[int]] = None,
        sample_row: Optional[pd.Series] = None,
        min_abs_points: Optional[float] = None,
        aggregate_label: Optional[str] = None):

        ADD_DUMMY_NO_ROW = True

        df = self._ensure_dataframe(df_in)
        if "condition" not in df.columns and "expr" in df.columns:
            df["condition"] = df["expr"]
        for c in ["condition", "points"]:
            if c not in df.columns:
                raise ValueError(f"Missing required column '{c}'")
        if "kind" not in df.columns:  
            df["kind"] = "uni"
        if "stage" not in df.columns: 
            df["stage"] = np.arange(len(df))

        def _n_feats(row):
            fs = row.get("features", None)
            if isinstance(fs, (list, tuple)):
                return len({self._strip_prefix(str(f)) for f in fs})
            toks = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", str(row.get("condition", "")))
            toks = [t for t in toks if t.upper() not in {"AND","OR","NOT","TRUE","FALSE","YES","NO"}]
            fams = []
            for t in toks:
                base = self._strip_prefix(t)
                fc = self._split_ohe(base)
                fams.append(fc[0] if fc else base)
            return len(set(fams))

        mask_eff_uni = (df["kind"].astype(str) == "pair") & (df.apply(_n_feats, axis=1) <= 1)
        df.loc[mask_eff_uni, "kind"] = "uni"


        df_uni  = df.loc[df["kind"].astype(str) == "uni"].copy()
        df_pair = df.loc[df["kind"].astype(str) == "pair"].copy() if include_pairs else pd.DataFrame(columns=df.columns)

        if sort_bins_by == "points_desc":
            df_uni  = df_uni.sort_values(["points", "condition"], ascending=[False, True], kind="mergesort")
            if not df_pair.empty:
                df_pair = df_pair.sort_values(["points", "condition"], ascending=[False, True], kind="mergesort")
        else:
            df_uni  = df_uni.sort_values("stage", ascending=True, kind="mergesort")
            if not df_pair.empty:
                df_pair = df_pair.sort_values("stage", ascending=True, kind="mergesort")

        vocab = self._collect_onehot_vocab(df)

        active_cat_by_fam: Dict[str, str] = {}
        if sample_row is not None:
            for fam, ohe_cols in (self.mapping or {}).items():
                for col in ohe_cols:
                    if col in sample_row and float(sample_row[col]) == 1.0:
                        active_cat_by_fam[fam] = col.replace(fam + "_", "", 1)

        aggregated: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        plain_rows: List[Tuple[str, float]] = []

        for _, row in df_uni.iterrows():
            pts = float(row["points"])
            feats = row.get("features", None)
            bounds = row.get("bounds", None)

            fam = None
            if isinstance(feats, (list, tuple)) and len(feats) == 1:
                fname = str(feats[0])
                fc = self._split_ohe(fname)
                if fc:
                    fam, cat = fc

            if fam and (self.mapping and fam in self.mapping):

                if fam not in aggregated:
                    aggregated[fam] = {"points": 0.0, "cats": {}, "stage_min": int(row["stage"])}
                aggregated[fam]["points"] += pts
                aggregated[fam]["cats"][cat] = pts
                aggregated[fam]["stage_min"] = min(aggregated[fam]["stage_min"], int(row["stage"]))
            
            elif bounds and isinstance(feats, (list, tuple)) and len(feats) == 1:
                var = str(feats[0])
                b = bounds.get(var, None)
                if b and ("lb" in b and "ub" in b):
                    lo = None if (b["lb"] is None or np.isneginf(b["lb"])) else float(b["lb"])
                    hi = None if (b["ub"] is None or np.isposinf(b["ub"])) else float(b["ub"])
                    lo_inc, hi_inc = True, True
                    base_var = self._strip_prefix(var)
                    key = f"NUM::{base_var}"
                    agg = aggregated.get(key)
                    if agg is None:
                        agg = aggregated[key] = {"var": base_var, "bins": [], "stage_min": int(row["stage"])}

                    # >>> PATCH: inverse-transform dei bound se c'è uno scaler
                    if self.scaler is not None:
                        vals, tags = [], []
                        if (lo is not None) and np.isfinite(lo):
                            vals.append(lo); tags.append("lo")
                        if (hi is not None) and np.isfinite(hi):
                            vals.append(hi); tags.append("hi")
                        if vals:
                            inv = self._inverse_feature_values(var, vals)
                            k = 0
                            if "lo" in tags:
                                lo = inv[k]; k += 1
                            if "hi" in tags:
                                hi = inv[k]

                    agg["bins"].append((lo, lo_inc, hi, hi_inc, pts))
                    agg["stage_min"] = min(agg["stage_min"], int(row["stage"]))

                else:
                    label = self._clean_expr_for_display(str(row["condition"]))
                    plain_rows.append({"label": label, "points": pts, "stage_min": int(row["stage"])})
            
            else:
                cond = str(row["condition"])
                cond = self._inverse_thresholds(cond)
                cond_no_pref = self._strip_prefix(cond)

                iv = self._parse_numeric_interval(cond_no_pref)
                if iv is not None and (feats is None or (isinstance(feats, (list, tuple)) and len(feats) <= 1)):
                    var, lo, lo_inc, hi, hi_inc = iv
                    base_var = self._strip_prefix(var)
                    key = f"NUM::{base_var}"
                    agg = aggregated.get(key)
                    if agg is None:
                        agg = aggregated[key] = {"var": base_var, "bins": [], "stage_min": int(row["stage"])}
                    agg["bins"].append((lo, lo_inc, hi, hi_inc, pts))
                    agg["stage_min"] = min(agg["stage_min"], int(row["stage"]))
                    continue

                cond_pretty = self._cond_pretty_all(cond_no_pref, vocab)
                label = self._clean_expr_for_display(cond_pretty)
                plain_rows.append({"label": label, "points": pts, "stage_min": int(row["stage"])})


        uni_rows = [["Bin", "Score"]]
        uni_pts  = []
        uni_items: List[Dict[str, Any]] = []

        is_local = sample_row is not None 

        for fam, info in aggregated.items():
            if "cats" not in info:
                continue
            fam_label = self._clean_identifier(fam)

            if is_local:
                active_cat = active_cat_by_fam.get(fam, None)
                if active_cat is not None and active_cat in info["cats"]:
                    pts = float(info["cats"][active_cat])
                    cat_label = self._clean_identifier(active_cat)
                    label = f"{fam_label} {cat_label}"
                    uni_items.append({
                        "label": label,
                        "points": pts,
                        "stage_min": int(info["stage_min"])
                    })
            else:
                for cat, pts in info["cats"].items():
                    cat_label = self._clean_identifier(cat)
                    uni_items.append({
                        "label": f"{fam_label} {cat_label}",
                        "points": float(pts),
                        "stage_min": int(info["stage_min"])
                    })
                    if ADD_DUMMY_NO_ROW:
                        uni_items.append({
                            "label": f"{fam_label} {cat_label} NO",
                            "points": 0.0,
                            "stage_min": int(info["stage_min"])
                        })

        for key, info in aggregated.items():
            if not key.startswith("NUM::"):
                continue
            var = info["var"]

            if is_local:
                bins_to_use = self._merge_active_bins(info["bins"])
            else:
                bins_to_use = self._partition_and_merge_intervals(info["bins"])

            for lo, lo_inc, hi, hi_inc, pts in bins_to_use:
                label = self._format_interval_label(var, lo, lo_inc, hi, hi_inc)
                uni_items.append({
                    "label": label,
                    "points": float(pts),
                    "stage_min": int(info["stage_min"])
                })


        uni_items.extend(plain_rows)

        df_uni_items = pd.DataFrame(uni_items)
        if df_uni_items.empty or "label" not in df_uni_items.columns:
            uni_items = []
        else:
            df_uni_items = (
                df_uni_items.groupby("label", as_index=False)
                .agg({"points": "sum", "stage_min": "min"})
            )
            uni_items = df_uni_items.to_dict("records")


        if sort_bins_by == "points_desc":
            uni_items.sort(key=lambda d: (-d["points"], d["stage_min"], d["label"]))
        else:
            uni_items.sort(key=lambda d: (d["stage_min"], -d["points"], d["label"]))

        if min_abs_points is not None:
            thr = float(min_abs_points)
            base_labels = {"[BASE]"}
            keep_uni, small_uni = [], []
            for it in uni_items:
                lbl = str(it.get("label", ""))
                if lbl in base_labels:
                    keep_uni.append(it)
                elif abs(float(it["points"])) >= thr:
                    keep_uni.append(it)
                else:
                    small_uni.append(it)
            if small_uni:
                small_sum = float(sum(x["points"] for x in small_uni))
                lab = aggregate_label or f"Other (|score| < {int(round(thr))})"
                keep_uni.append({
                    "label": lab,
                    "points": small_sum,
                    "stage_min": max([k["stage_min"] for k in keep_uni], default=10**9) + 1
                })
            if sort_bins_by == "points_desc":
                keep_uni.sort(key=lambda d: (-d["points"], d["stage_min"], d["label"]))
            else:
                keep_uni.sort(key=lambda d: (d["stage_min"], -d["points"], d["label"]))
            uni_items = keep_uni


        for it in uni_items:
            uni_rows.append([it["label"], f"{int(round(it['points'])):+d}"])
            uni_pts.append(it["points"])

        pair_rows, pair_pts = [], []
        if include_pairs and not df_pair.empty:
            pair_rows.append(["Bin", "Score"])
            for _, row in df_pair.iterrows():
                pts = float(row["points"])
                cond = self._inverse_thresholds(str(row["condition"]))
                cond_no_pref = self._strip_prefix(cond)
                cond_pretty  = self._cond_pretty_all(cond_no_pref, vocab)
                cond_display = self._clean_expr_for_display(cond_pretty)
                pair_rows.append([cond_display, f"{int(round(pts)):+d}"])
                pair_pts.append(pts)
        
        pair_rows, pair_pts = [], []
        if include_pairs and not df_pair.empty:
            pair_items = []
            for _, row in df_pair.iterrows():
                pts = float(row["points"])
                cond = self._inverse_thresholds(str(row["condition"]))
                cond_no_pref = self._strip_prefix(cond)
                cond_pretty  = self._cond_pretty_all(cond_no_pref, vocab)
                cond_display = self._clean_expr_for_display(cond_pretty)
                pair_items.append({
                    "label": cond_display,
                    "points": pts,
                    "stage_min": int(row.get("stage", 10**6))
                })

            if sort_bins_by == "points_desc":
                pair_items.sort(key=lambda d: (-d["points"], d["stage_min"], d["label"]))
            else:
                pair_items.sort(key=lambda d: (d["stage_min"], -d["points"], d["label"]))

            if min_abs_points is not None:
                thr = float(min_abs_points)
                keep_pair = [it for it in pair_items if abs(float(it["points"])) >= thr]
                small_pair = [it for it in pair_items if abs(float(it["points"])) < thr]
                if small_pair:
                    small_sum = float(sum(x["points"] for x in small_pair))
                    lab = aggregate_label or f"Other (|score| < {int(round(thr))})"
                    keep_pair.append({
                        "label": lab,
                        "points": small_sum,
                        "stage_min": max([k["stage_min"] for k in keep_pair], default=10**9) + 1
                    })
                if sort_bins_by == "points_desc":
                    keep_pair.sort(key=lambda d: (-d["points"], d["stage_min"], d["label"]))
                else:
                    keep_pair.sort(key=lambda d: (d["stage_min"], -d["points"], d["label"]))
                pair_items = keep_pair


            if pair_items:
                pair_rows.append(["Bin", "Score"])
                for it in pair_items:
                    pair_rows.append([it["label"], f"{int(round(it['points'])):+d}"])
                    pair_pts.append(it["points"])

        return (uni_rows, uni_pts), (pair_rows, pair_pts)
    
    def render_interactive_html_from_template(
        self,
        *,
        sample=None,
        include_pairs: bool = True,
        sort_bins_by: str = "points_desc",
        title: str = "Scorecard",
        template_path: str = "scorecard_template.html",
        savepath: str = "scorecard.html",
        band_probs: Optional[List[float]] = None,
        band_labels: Optional[List[str]] = None,
        topk_factors: int = 5
    ) -> str:
        import json, html, os
        from pathlib import Path
        import pandas as pd


        df_global = self.global_table()
        (g_uni_rows, g_uni_pts), (g_pair_rows, g_pair_pts) = self._build_scorecard_table_data(
            df_global,
            include_pairs=include_pairs,
            sort_bins_by=sort_bins_by,
            selected_rule_ids=None,
            sample_row=None
        )

        def rows_to_obj(rows, pts):
            if not rows:
                return []
            out = []
            for (lab, sc), p in zip(rows[1:], pts):
                out.append({"label": str(lab), "score": str(sc), "pts": float(p)})
            return out

        data_global = {
            "uni": rows_to_obj(g_uni_rows, g_uni_pts),
            "pair": rows_to_obj(g_pair_rows, g_pair_pts) if g_pair_rows else []
        }

        data_local = None
        local_summary = None
        if sample is not None:
            if isinstance(sample, dict):
                sample_row = pd.Series(sample)
            elif isinstance(sample, pd.DataFrame):
                if sample.shape[0] != 1:
                    raise ValueError("In 'LOCAL', sample deve avere esattamente 1 riga.")
                sample_row = sample.iloc[0]
            else:
                sample_row = sample

            df_local, S_points, proba, yhat = self.local_table(sample_row)
            (l_uni_rows, l_uni_pts), (l_pair_rows, l_pair_pts) = self._build_scorecard_table_data(
                df_local.rename(columns={"expr": "condition"}),
                include_pairs=include_pairs,
                sort_bins_by=sort_bins_by,
                selected_rule_ids=None,
                sample_row=sample_row
            )
            data_local = {
                "uni": rows_to_obj(l_uni_rows, l_uni_pts),
                "pair": rows_to_obj(l_pair_rows, l_pair_pts) if l_pair_rows else []
            }
            thr = int(round(self.scorecard.offset))
            local_summary = {
                "total_points": float(S_points),
                "base_points": float(self.scorecard.base_points),
                "threshold": float(thr),
                "decision": f"{int(S_points)} {'>=' if S_points >= thr else '<'} {thr}",
                "pred_class": int(1 if S_points >= thr else 0),
                "proba_p1": float(proba[1]),
                "proba_p0": float(proba[0]),
            }


        risk_bands = self.compute_risk_bands(probs=band_probs, labels=band_labels)
        factors = self.compute_factors_share(top_k=topk_factors)

        tpath = Path(template_path)
        if not tpath.exists():
            base = Path(os.path.dirname(__file__)) if "__file__" in globals() else Path.cwd()
            alt = base / template_path
            if alt.exists():
                tpath = alt
            else:
                raise FileNotFoundError(f"Template non trovato: {template_path}")

        tpl = tpath.read_text(encoding="utf-8")

        replacements = {
            "{{TITLE}}": html.escape(title),
            "{{HAS_LOCAL}}": "true" if data_local is not None else "false",
            "{{DATA_GLOBAL_JSON}}": json.dumps(data_global, ensure_ascii=False),
            "{{DATA_LOCAL_JSON}}": json.dumps(data_local, ensure_ascii=False) if data_local is not None else "null",
            "{{LOCAL_SUMMARY_JSON}}": json.dumps(local_summary, ensure_ascii=False) if local_summary is not None else "null",

            "{{RISK_BANDS_JSON}}": json.dumps(risk_bands, ensure_ascii=False),
            "{{FACTORS_JSON}}": json.dumps(factors, ensure_ascii=False),
        }

        for k, v in replacements.items():
            tpl = tpl.replace(k, v)

        Path(savepath).write_text(tpl, encoding="utf-8")
        return savepath

    def render(
        self,
        mode: str = "global", # "global" | "local"
        sample: Optional[Union[pd.Series, pd.DataFrame, Dict[str, Any]]] = None,
        *,
        include_pairs: bool = True,
        sort_bins_by: str = "points_desc", # "stage" | "points_desc"
        selected_rule_ids: Optional[Iterable[int]] = None,
        title: Optional[str] = None,
        dpi: int = 150,
        savepath: Optional[str] = "scorecard_table.png",
        cmap_name: str = "BrBG",

        wrap_width: Optional[int] = None,
        base_row_h: Optional[float] = None,
        row_gap: Optional[float] = None,
        min_row_inch: Optional[float] = None,
        cell_pad: Optional[float] = None,
        col_widths: Optional[Tuple[float, float]] = None,
        font_min: Optional[int] = None,
        font_max: Optional[int] = None,
        max_fig_height: Optional[float] = None,
        min_abs_points: Optional[float] = None,
        aggregate_label: Optional[str] = None,
    ) -> str:

        sc = self.scorecard

        prof = self._layout_profile(mode)

        wrap_width   = prof["wrap_width"]   if wrap_width   is None else wrap_width
        base_row_h   = prof["base_row_h"]   if base_row_h   is None else base_row_h
        row_gap      = prof["row_gap"]      if row_gap      is None else row_gap
        min_row_inch = prof["min_row_inch"] if min_row_inch is None else min_row_inch
        cell_pad     = prof["cell_pad"]     if cell_pad     is None else cell_pad
        col_widths   = prof["col_widths"]   if col_widths   is None else col_widths
        font_min     = prof["font_min"]     if font_min     is None else font_min
        font_max     = prof["font_max"]     if font_max     is None else font_max
        max_fig_hard = prof["max_fig_height"] if max_fig_height is None else max_fig_height

        annotate_total = False
        points_threshold = int(round(sc.offset))
        total_points: Optional[float] = None

        if mode == "global":
            df_in = self.global_table()
        elif mode == "local":
            if sample is None:
                raise ValueError("In 'local' mode, 'sample' dev'essere fornito.")
            df_local, S_points, proba, yhat = self.local_table(sample)
            df_in = df_local.rename(columns={"expr": "condition"})
            annotate_total = True
            total_points = float(S_points)
        else:
            raise ValueError("mode dev'essere 'global' o 'local'.")

        sample_row = None
        if mode == "local":
            if isinstance(sample, dict): sample_row = pd.Series(sample)
            elif isinstance(sample, pd.DataFrame): sample_row = sample.iloc[0]
            elif isinstance(sample, pd.Series): sample_row = sample

        (uni_rows, uni_pts), (pair_rows, pair_pts) = self._build_scorecard_table_data(
            df_in,
            include_pairs=include_pairs,
            sort_bins_by=sort_bins_by,
            selected_rule_ids=selected_rule_ids,
            sample_row=sample_row,
            aggregate_label=aggregate_label,
            min_abs_points=min_abs_points
        )

        def _soft_break_tokens(s: str, every: int = 18) -> str:
            ZWSP = "\u200b"
            pat = re.compile(rf"(\S{{{every}}})(?=\S)")
            return pat.sub(lambda m: m.group(1) + ZWSP, s)
        
                
        def _wrap_by_pixels(text: str, max_w_px: float, fontsize: float, renderer, soft_every: int = 24):
            ZWSP = "\u200b"
            pat = re.compile(rf"(\S{{{soft_every}}})(?=\S)")
            s = pat.sub(lambda m: m.group(1) + ZWSP, str(text))

            words = s.split()
            if not words:
                return [""]

            lines = [words[0]]
            for w in words[1:]:
                cand = lines[-1] + " " + w
                t = Text(0, 0, cand, fontsize=fontsize)
                t.set_figure(fig)
                bb = t.get_window_extent(renderer=renderer, dpi=fig.dpi)
                if bb.width <= max_w_px:
                    lines[-1] = cand
                else:
                    lines.append(w)
            return lines

        

        uni_linecounts  = [1] * max(0, len(uni_rows)  - 1)
        pair_linecounts = [1] * max(0, (len(pair_rows) - 1) if pair_rows else 0)

        uni_units  = 1 + sum(uni_linecounts)
        pair_units = (1 + sum(pair_linecounts)) if pair_rows else 0
        total_units_all = uni_units + pair_units
        n_uni  = len(uni_rows)
        n_pair = len(pair_rows) if pair_rows else 0
        n_rows_grand_total = n_uni + n_pair

        margin = 0.6 if title else 0.35

        extra_top = 0.0
        if annotate_total:
            TOP_PAD = 0.02
            BAR_H   = 0.08
            GAP     = 0.02
            extra_top = TOP_PAD + BAR_H + GAP

        height_theoretical = max(
            min_row_inch * (1 + n_rows_grand_total),
            base_row_h * total_units_all + margin + extra_top + (row_gap if pair_rows else 0.0),
            2.6,
        )


        if height_theoretical > max_fig_hard:
            scale = max(
                0.25,
                (max_fig_hard - margin - (row_gap if pair_rows else 0.0)) / (base_row_h * total_units_all)
            )
            base_row_h_eff = base_row_h * scale
            height = max_fig_hard
        else:
            base_row_h_eff = base_row_h
            height = height_theoretical

        fig, ax = plt.subplots(figsize=(10, height), dpi=dpi)
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=15, loc="center", pad=10)

        y_cursor = 1.0 - extra_top

        def _make_norm_for(pts):
            pts = [float(p) for p in pts if p is not None]
            if not pts:
                return None, None
            pmin, pmax = min(pts), max(pts)
            if pmin < 0 < pmax:
                vmax = max(abs(pmin), abs(pmax))
                norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
            else:
                norm = colors.Normalize(vmin=pmin, vmax=pmax)
            return norm, colormaps.get_cmap(cmap_name)


        def _fg_for_bg(rgba):
            r, g, b, _ = rgba
            L = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return "black" if L > 0.5 else "white"

        def _compute_fontsize():
            raw = font_max - 0.003 * n_rows_grand_total
            return max(font_min, min(font_max, raw))

        def _add_table(rows, pts_per_row, subtitle: Optional[str] = None):
            nonlocal y_cursor
            if not rows:
                return

            if fig.canvas.get_renderer() is None:
                fig.canvas.draw()
            renderer = fig.canvas.get_renderer()

            norm, cmap = _make_norm_for(pts_per_row)

            if subtitle:
                ax.text(0.01, y_cursor, subtitle, fontsize=10, va="top", ha="left")
                y_cursor -= 0.06

            fs_base = _compute_fontsize()

            fig_w_in = fig.get_size_inches()[0]
            fig_w_px = fig_w_in * dpi
            ax_pos = ax.get_position()
            axes_w_px = fig_w_px * ax_pos.width
            table_w_px = axes_w_px * 0.98
            col0_w_px = table_w_px * float(col_widths[0])
            inner_pad_px = 8.0
            max_w_px = max(10.0, col0_w_px - inner_pad_px)

            wrapped_rows = [rows[0][:]]
            linecounts = [1]

            agg_flags = []
            for cond, score in rows[1:]:
                is_agg = (
                    (aggregate_label is not None and cond.strip() == aggregate_label.strip())
                    or cond.strip().startswith("Other")
                )
                agg_flags.append(is_agg)

                lines = _wrap_by_pixels(cond, max_w_px, fs_base, renderer)
                MAX_LINES = 3
                if len(lines) > MAX_LINES:
                    lines = lines[:MAX_LINES]
                    lines[-1] = lines[-1].rstrip() + "…"
                wrapped_rows.append(["\n".join(lines), score])
                linecounts.append(max(1, len(lines)))

            row_units = [1] + linecounts[1:]
            total_units = sum(row_units)
            table_h = base_row_h_eff * total_units
            col_count = len(rows[0])

            table = ax.table(
                cellText=wrapped_rows,
                colWidths=list(col_widths),
                cellLoc="left",
                loc="upper left",
                bbox=[0.01, y_cursor - table_h, 0.98, table_h],
            )

            AGG_BG = "#b3b1b1"
            score_col_idx = col_count - 1

            for r in range(len(wrapped_rows)):
                units = row_units[r]
                h = (units / total_units) * table_h

                raw_label = rows[r][0] if r > 0 else ""
                is_agg_row = (
                    isinstance(raw_label, str)
                    and (
                        raw_label.strip() == (aggregate_label or "").strip()
                        or raw_label.lower().startswith("other")
                        or raw_label.lower().startswith("altre")
                    )
                )

                for c in range(col_count):
                    cell = table[r, c]
                    cell.set_height(h)
                    cell.set_linewidth(0.6 if r else 1.0)

                    txt = cell.get_text()
                    txt.set_wrap(False)
                    txt.set_fontsize((fs_base + 1) if r == 0 else fs_base)

                    if r == 0:
                        cell.set_text_props(weight="bold")
                        if c == score_col_idx:
                            txt.set_ha("right")
                        continue

                    if c == 0:
                        txt.set_ha("left")
                    if c == score_col_idx:
                        txt.set_ha("right")
                        txt.set_fontfamily("monospace")

                    if is_agg_row and c == 0:
                        cell.set_facecolor(AGG_BG)
                        txt.set_color("black")
                    elif c == score_col_idx and norm is not None and cmap is not None:
                        p = pts_per_row[r - 1]
                        if p is not None:
                            rgba = cmap(norm(float(p)))
                            cell.set_facecolor(rgba)
                            txt.set_color(_fg_for_bg(rgba))

            y_cursor -= (table_h + row_gap)



        Cell.PAD = cell_pad

        _add_table(uni_rows, uni_pts, subtitle="Univariate rules")
        if pair_rows:
            _add_table(pair_rows, pair_pts, subtitle="Pairwise rules")
        

        if annotate_total and (total_points is not None):
            pred_class = 1 if total_points >= points_threshold else 0
            status_color = "#2ecc71" if pred_class == 1 else "#e74c3c"

            proba = self._predict_proba_from_score(
                [total_points], factor=sc.factor, offset=sc.offset
            )[0]
            p1 = float(proba[1])
            pct_txt = f"{int(round(p1*100))}%"

            left, right = 0.02, 0.98
            width = right - left
            TOP_PAD, BAR_H = 0.02, 0.08
            bottom = 1 - TOP_PAD - BAR_H
            center_y = bottom + BAR_H/2

            from matplotlib.patches import Rectangle, FancyBboxPatch

            ax.add_patch(FancyBboxPatch(
                (left, bottom), width, BAR_H,
                boxstyle="round,pad=0.01",
                facecolor=status_color, edgecolor=status_color,
                alpha=0.15, transform=ax.transAxes, zorder=10, clip_on=False
            ))

            ax.text(left + 0.01*width, center_y,
                    f"PREDICTION: CLASS {pred_class}",
                    fontsize=13, weight='bold', color=status_color,
                    va='center', ha='left', transform=ax.transAxes, zorder=11)

            ax.text(left + 0.35*width, center_y,
                    f"Score: {int(total_points)} | Threshold: {points_threshold}",
                    fontsize=11, weight='bold', color='#2c3e50',
                    va='center', ha='left', transform=ax.transAxes, zorder=11)

            bar_w = 0.18*width
            bar_x = left + 0.7*width
            prog_h = 0.35*BAR_H
            prog_y = center_y - prog_h/2

            ax.add_patch(Rectangle(
                (bar_x, prog_y), bar_w, prog_h,
                facecolor='#ecf0f1', edgecolor='#95a5a6', linewidth=1,
                transform=ax.transAxes, zorder=11
            ))
            ax.add_patch(Rectangle(
                (bar_x, prog_y), bar_w * p1, prog_h,
                facecolor=status_color, edgecolor='none',
                transform=ax.transAxes, zorder=12
            ))
            ax.text(bar_x + bar_w/2, center_y, pct_txt,
                    fontsize=8, color='black', weight='bold',
                    va='center', ha='center', transform=ax.transAxes, zorder=13)
                    
        if savepath is None:
            savepath = "scorecard_table.png"
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
        return savepath