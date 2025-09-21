import os

import re
from typing import Union, List, Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utils import load_preprocess, get_available_datasets


class RuleExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts rules from DecisionTree/RandomForest and evaluates them as binary features (vectorized).
    methods: 'decision_tree' | 'random_forest'
    """

    def __init__(self,
                 method: str = 'decision_tree',
                 max_depth: int = 3,
                 n_estimators: int = 10,
                 only_positive: bool = False,
                 dtype=np.uint8,
                 feature_name_map: Optional[Dict[str, str]] = None,
                 string_normalizer: Optional[callable] = None,
                 auto_map_by_index: bool = True):
        self.method = method
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.only_positive = only_positive
        self.dtype = dtype
        self.feature_name_map = feature_name_map or {}
        self.string_normalizer = string_normalizer
        self.auto_map_by_index = auto_map_by_index

        self.feature_names: List[str] = []
        self.model = None
        self.active_rules_: List[dict] = []
        self.rule_names_: List[str] = []
        self.meta_: List[dict] = [] # leaf info: prediction, prob_1
        self.tree_dict_ = None
        

    # -------------------- utils --------------------
    @staticmethod
    def _to_numpy(X, y=None):
        X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
        if y is None:
            return X_np
        y_np = y.to_numpy().ravel() if isinstance(y, (pd.Series, pd.DataFrame)) else np.asarray(y).ravel()
        return X_np, y_np

    @staticmethod
    def _fmt_float(v: float) -> str:
        return f"{float(v):.6g}"

    def _resolve_display_name(self, cond: dict) -> str:
        feat = str(cond.get('feature_name'))
        if feat in self.feature_name_map:
            return self.feature_name_map[feat]
        if feat in self.feature_names:
            return feat
        idx = cond.get('feature_idx')
        if idx is not None and 0 <= int(idx) < len(self.feature_names):
            return self.feature_names[int(idx)]
        m = re.match(r'^X_(\d+)$', feat)
        if m:
            i = int(m.group(1))
            if 0 <= i < len(self.feature_names):
                return self.feature_names[i]
        return feat

    def _resolve_col_in_df(self, df: pd.DataFrame, cond: dict) -> str:
        feat = str(cond.get('feature_name'))
        mapped = self.feature_name_map.get(feat)
        if mapped in df.columns:
            return mapped
        if feat in df.columns:
            return feat
        if not self.auto_map_by_index:
            raise KeyError(f"Colonna per '{feat}' non trovata (auto_map_by_index=False).")
        idx = cond.get('feature_idx')
        if idx is not None and 0 <= int(idx) < len(df.columns):
            return df.columns[int(idx)]
        m = re.match(r'^X_(\d+)$', feat)
        if m:
            i = int(m.group(1))
            if 0 <= i < len(df.columns):
                return df.columns[i]
        raise KeyError(f"Impossibile risolvere la feature '{feat}' (override/nome/idx/X_i).")

    def _apply_op(self, series: pd.Series, op: str, thr: Any, is_cat: bool) -> np.ndarray:
        norm = self.string_normalizer
        if is_cat:
            if norm is None and isinstance(thr, str) and thr != thr.strip():
                norm = str.strip
            if norm is not None:
                lhs = series.astype(object).map(lambda x: norm(str(x)) if x is not None and x is not np.nan else x)
                rhs = norm(str(thr))
            else:
                lhs, rhs = series, thr
            return (lhs == rhs).to_numpy() if op == '==' else (lhs != rhs).to_numpy()

        arr = pd.to_numeric(series, errors='coerce').to_numpy()
        t = float(thr)
        if op == '<=':
            return np.less_equal(arr, t)
        if op == '>':
            return np.greater(arr, t)
        raise ValueError(f"Operatore sconosciuto: {op}")

    def _rule_name(self, rule: dict) -> str:
        parts = []
        for c in rule['conditions']:
            disp = self._resolve_display_name(c)
            thr = c['threshold']
            t = thr if isinstance(thr, str) else self._fmt_float(thr)

            if isinstance(thr, str) and (thr.strip() != thr or ' ' in thr):
                t = repr(thr)
            parts.append(f"{disp} {c['op']} {t}")
        return " & ".join(parts) if parts else "(TRUE)"

    # -------------------- rule extraction --------------------
    def _extract_rules_from_sklearn_tree(self, est, feature_names: List[str]) -> List[dict]:
        """Ritorna una lista di regole con conditions + meta (prediction, prob_1)."""
        from sklearn.tree import _tree
        tree_ = est.tree_
        classes = getattr(est, "classes_", None)
        classes = classes if classes is None else np.asarray(classes)

        rules = []

        def recurse(node, path):
            fid = tree_.feature[node]
            if fid != _tree.TREE_UNDEFINED:  # internal node
                thr = float(tree_.threshold[node])
                cond_left = {'feature_name': feature_names[fid], 'feature_idx': int(fid),
                             'is_categorical': False, 'op': '<=', 'threshold': thr}
                cond_right = {'feature_name': feature_names[fid], 'feature_idx': int(fid),
                              'is_categorical': False, 'op': '>', 'threshold': thr}
                recurse(tree_.children_left[node], path + [cond_left])
                recurse(tree_.children_right[node], path + [cond_right])
            else:
                # leaf: pred + prob_1 (if binary and class '1' present)
                counts = tree_.value[node][0]
                pred_idx = int(np.argmax(counts))
                pred = int(classes[pred_idx]) if classes is not None else pred_idx
                p1 = None
                if classes is not None and len(classes) >= 2 and 1 in classes:
                    denom = counts.sum()
                    p1 = float(counts[int(np.where(classes == 1)[0][0])] / denom) if denom > 0 else None
                rules.append({'conditions': path.copy(), 'prediction': pred, 'prob_1': p1})

        recurse(0, [])
        return rules

    '''    # -------------------- RuleTree rule extraction --------------------
    def _extract_leaf_rules_from_ruletree(self, tree_dict: dict) -> List[dict]:
        nodes = tree_dict.get('nodes', [])
        id2node = {n['node_id']: n for n in nodes}
        children = {n.get('left_node') for n in nodes if not n.get('is_leaf')} |  {n.get('right_node') for n in nodes if not n.get('is_leaf')}
        root_id = next((n['node_id'] for n in nodes if n['node_id'] not in children), None)
        if root_id is None:
            raise ValueError("Radice non trovata in to_dict().")

        rules = []

        def walk(nid, path):
            node = id2node[nid]
            if node.get('is_leaf'):
                prob = node.get('prediction_probability')
                p1 = float(prob[-1]) if prob is not None else None
                rules.append({
                    'conditions': path.copy(),
                    'prediction': int(node.get('prediction')) if node.get('prediction') is not None else None,
                    'prob_1': p1,
                })
                return
            is_cat = bool(node.get('is_categorical'))
            thr = node.get('threshold')
            feat_idx = int(node['feature_idx']) if node.get('feature_idx') is not None else None
            feat_name = str(node.get('feature_name'))
            if is_cat:
                left =  {'feature_name': feat_name, 'feature_idx': feat_idx, 'is_categorical': True,  'op': '==', 'threshold': str(thr)}
                right = {'feature_name': feat_name, 'feature_idx': feat_idx, 'is_categorical': True,  'op': '!=', 'threshold': str(thr)}
            else:
                thr = float(thr)
                left =  {'feature_name': feat_name, 'feature_idx': feat_idx, 'is_categorical': False, 'op': '<=', 'threshold': thr}
                right = {'feature_name': feat_name, 'feature_idx': feat_idx, 'is_categorical': False, 'op': '>',  'threshold': thr}
            if node.get('left_node') is not None:
                walk(node['left_node'], path + [left])
            if node.get('right_node') is not None:
                walk(node['right_node'], path + [right])

        walk(root_id, [])
        return rules'''

    # -------------------- API --------------------
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "RuleExtractor":
        self.feature_names = list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(X.shape[1])]
        method = self.method.lower()

        if method == 'decision_tree':
            self.model = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
            self.model.fit(X, y)
            all_rules = self._extract_rules_from_sklearn_tree(self.model, self.feature_names)

        elif method == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth, n_jobs=-1, random_state=42
            )
            self.model.fit(X, y)
            all_rules = []
            for est in self.model.estimators_:
                all_rules.extend(self._extract_rules_from_sklearn_tree(est, self.feature_names))
        else:
            raise ValueError("Unsupported method. Choose from: 'decision_tree' or 'random_forest'")

        if self.only_positive:
            all_rules = [r for r in all_rules if r.get('prediction') == 1]

        self.active_rules_ = all_rules
        self.rule_names_ = [self._rule_name(r) for r in all_rules]
        self.meta_ = [
            {'name': n, 'prediction': r.get('prediction'), 'prob_1': r.get('prob_1')}
            for n, r in zip(self.rule_names_, all_rules)
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.active_rules_:
            raise ValueError("Model not fitted. Call fit() first.")

        out_cols = []
        for rule, cname in zip(self.active_rules_, self.rule_names_):
            mask = np.ones(len(X), dtype=bool)
            for c in rule['conditions']:
                col = self._resolve_col_in_df(X, c)
                mask &= self._apply_op(X[col], c['op'], c['threshold'], bool(c.get('is_categorical')))
                if not mask.any(): 
                    break
            out_cols.append(pd.Series(mask.astype(self.dtype), index=X.index, name=cname))

        return pd.concat(out_cols, axis=1) if out_cols else pd.DataFrame(index=X.index, dtype=self.dtype)

def load_data(name):
    datasets_info = get_available_datasets().get(name)
    class_col = datasets_info['class_col']
    positive_class = datasets_info['positive_class']

    if datasets_info is None:
        raise ValueError(f"Dataset '{name}' not found.")
    
    df = pd.read_csv(datasets_info['path'])
    if datasets_info['idCol'] is not None:
        df = df.drop(columns=[datasets_info['idCol']])
        print(f"ID column '{datasets_info['idCol']}' dropped.")
    df[class_col] = df[class_col].astype(str).str.strip()
    class_unique = df[class_col].unique()
    assert len(class_unique) == 2, f"Target ha più di due classi: {class_unique}"
    
    if positive_class is not None:
        mapping = {c: int(c == positive_class) for c in class_unique}
    else:
        mapping = {cls: idx for idx, cls in enumerate(sorted(class_unique))}

    df[class_col] = df[class_col].map(mapping)
    
    X = df.drop(columns=class_col)
    y = df[class_col]
    return X, y

if __name__ == "__main__":
    
    data_dir = "DATA/"
    
    os.makedirs(data_dir+"DT", exist_ok=True)
    os.makedirs(data_dir+"RT", exist_ok=True)
    os.makedirs(data_dir+"RF", exist_ok=True)

    all_datasets = get_available_datasets()
    summary = []

    for name, info in all_datasets.items():
        X, y = load_data(name)
        X_dummies = pd.get_dummies(X, drop_first=True)

        extractor_dt = RuleExtractor(method='decision_tree', max_depth=2)
        extractor_dt.fit(X_dummies, y)
        X_rules_dt = extractor_dt.transform(X_dummies)
        full_df_dt = pd.concat([X_rules_dt, y.rename("class")], axis=1)
        full_df_dt.to_csv(f"{data_dir}DT/{name}_rules.csv", index=False)

        extractor_rf = RuleExtractor(method='random_forest', max_depth=2, n_estimators=5)
        extractor_rf.fit(X_dummies, y)
        X_rules_rf = extractor_rf.transform(X_dummies)
        full_df_rf = pd.concat([X_rules_rf, y.rename("class")], axis=1)
        full_df_rf.to_csv(f"{data_dir}RF/{name}_rules.csv", index=False)

        summary.append({
            "dataset": name,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_rules_dt": X_rules_dt.shape[1],
            "n_rules_rf": X_rules_rf.shape[1],
        })
    print(summary)