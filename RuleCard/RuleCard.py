from __future__ import annotations
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder

import time
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from AdditiveModel import AdditiveModel
from Scorecard import Scorecard
try:
    from ScorecardVisualizer import ScorecardVisualizer
except Exception:
    ScorecardVisualizer = None

try:
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.utils.validation import check_is_fitted
except Exception:
    BaseEstimator = object
    ClassifierMixin = object
    def check_is_fitted(estimator, attributes=None):
        attrs = attributes or ['scorecard_']
        for a in (attrs if isinstance(attrs, (list, tuple, set)) else [attrs]):
            if not hasattr(estimator, a):
                raise AttributeError("This RuleCardClassifier instance is not fitted yet.")


def _clean_column_names(columns):
    seen = {}
    cleaned = []
    for col in columns:
        new_name = re.sub(r'[^A-Za-z0-9_]', '_', col)
        new_name = re.sub(r'_+', '_', new_name)
        new_name = new_name.strip('_')
        if new_name in seen:
            seen[new_name] += 1
            new_name = f"{new_name}_{seen[new_name]}"
        else:
            seen[new_name] = 0
        cleaned.append(new_name)
    return cleaned

def get_feature_importance(X_train, y_train):
    selector = SelectKBest(score_func=chi2, k='all')
    selector.fit(X_train, y_train)
    scores = selector.scores_
    feature_names = X_train.columns
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': scores
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    feature_order = importance['feature'].tolist()
    return feature_order, importance


def load_data(
    file_path: str,
    sep: str = ",",
    class_col: str | None = None,
    positive_class: str | None = None,
    id_col: str | None = None
):
    """
    Load and preprocess a dataset:
    - Cleans column names
    - Handles target column (`class_col`) and optional ID column (`id_col`)
    - Numeric missing values -> arbitrary out-of-range value
    - Categorical missing values -> "Unknown"
    - One-hot encoding on categorical features

    Parameters:
    - file_path: path to the CSV file
    - sep: separator
    - class_col: name of the target column (if provided, excluded from preprocessing)
    - positive_class: value to be mapped to 1 (other classes to 0). 
                      If None → multi-class mapping
    - id_col: identifier column to exclude from preprocessing

    Returns:
    - df_final: preprocessed DataFrame (features only)
    - y: Series with target mapped to integers
    - ids: Series with IDs (if `id_col` is specified), otherwise None
    - col_mapping: dict {original categorical column: one-hot columns}
    - class_mapping: dict {original class value: int}
    """
    df = pd.read_csv(file_path, sep=sep)
    df.columns = _clean_column_names(df.columns)

    ids = None
    if id_col is not None:
        if id_col not in df.columns:
            raise ValueError(f"Column '{id_col}' does not exist in the dataset.")
        ids = df[id_col]
        df = df.drop(columns=[id_col])

    y = None
    class_mapping = None
    if class_col is not None:
        if class_col not in df.columns:
            raise ValueError(f"Column '{class_col}' does not exist in the dataset.")
        
        df[class_col] = df[class_col].astype(str).str.strip().str.lower()
        class_unique = df[class_col].unique()

        if positive_class is not None:
            pos_norm = positive_class.strip().lower()
            class_mapping = {c: 1 if c == pos_norm else 0 for c in class_unique}
        else:
            class_mapping = {cls: idx for idx, cls in enumerate(sorted(class_unique))}

        y = df[class_col].map(class_mapping)
        df = df.drop(columns=[class_col])

    obj_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if class_col in obj_cols:
        obj_cols.remove(class_col)

    for col in obj_cols:
        s = df[col].astype(str).str.strip()
        s = s.mask(s == '?', np.nan)
        df[col] = s
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in num_cols:
        if df[col].isnull().any():
            min_val, max_val = df[col].min(), df[col].max()
            if pd.notnull(max_val) and pd.notnull(min_val):
                out_of_range_val = max_val + (max_val - min_val + 1)
            else:
                out_of_range_val = 999999
            df[col] = df[col].fillna(out_of_range_val)

    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_array = encoder.fit_transform(df[cat_cols])

    encoded_cols = _clean_column_names(encoder.get_feature_names_out(cat_cols))
    df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

    df_final = pd.concat([df[num_cols], df_encoded], axis=1)

    col_mapping = {
        col: [c for c in encoded_cols if c.startswith(col + "_")]
        for col in cat_cols
    }

    return df_final, y, ids, col_mapping, class_mapping





# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
class RuleCardClassifier(BaseEstimator, ClassifierMixin):
    """
    Trains an additive model to discover rules and converts it into a scorecard 
    with points, optional pruning and optional probability calibration.

    Parameters
    ----------
    model__* : any
        Keyword arguments forwarded to :class:`AdditiveModel.AdditiveModel`.
        Use the `model__` prefix to tune hyper-parameters via GridSearchCV.

    min_support : int | float, default=0.005
        Minimum support to keep a rule when extracting from the additive model.
        If float in (0,1), it's interpreted as a fraction of the training set.

    max_rules : int | None, default=None
        If provided, keeps only the top-`max_rules` rules by absolute points
        (delegated to :meth:`Scorecard.prune_rules`).

    PDO : int, default=50
        Points to double the odds for the score scaling.

    score0 : float, default=0.0
        Base score anchor (in points).

    odds0 : float | None, default=None
        Base odds anchor. If None, it will be inferred from the training base rate.

    bounds_col : str, default='bounds'
        Column name containing rule bounds in the rules table.

    on_missing : {'ignore','error'}, default='ignore'
        What to do if a feature needed by some rule is missing at predict time.

    return_sparse : bool, default=True
        If True, internal activation matrices are sparse.

    calibrate : bool, default=False
        Whether to fit a probability calibrator over the score.

    calibrator__method : {'platt','isotonic'}, default='platt'
    calibrator__n_splits : int, default=5
    calibrator__random_state : int, default=42
    calibrator__class_weight : dict | 'balanced' | None, default=None
    calibrator__max_iter : int, default=2000
        Calibration options forwarded to :meth:`Scorecard.fit_calibrator`.

    predict_threshold : float, default=0.5
        Threshold in probability space for :meth:`predict`.

    Attributes
    ----------
    additive_model_ : AdditiveModel
        Fitted additive model.

    scorecard_ : Scorecard
        Fitted scorecard converted from the additive model.

    classes_ : ndarray of shape (2,)
        Class labels seen during fit (expects binary classification).

    feature_names_in_ : ndarray of shape (n_features,)
        Feature names learned by the additive model.

    training_info_ : dict
        Basic fit diagnostics (n_samples, class_balance, times, etc.).
    """
    # ------------------------------ init -----------------------------------
    def __init__(
        self,
        *,
        # Scorecard extraction options
        min_support: Union[int, float] = 0.005,
        max_rules: Optional[int] = None,
        PDO: int = 50,
        score0: float = 0.0,
        odds0: Optional[float] = None,
        bounds_col: str = "bounds",
        on_missing: str = "ignore",
        return_sparse: bool = True,
        # Calibration
        calibrate: bool = False,
        calibrator__method: str = "platt",
        calibrator__n_splits: int = 5,
        calibrator__random_state: int = 42,
        calibrator__class_weight: Optional[Union[str, dict]] = None,
        calibrator__max_iter: int = 2000,
        # Prediction
        predict_threshold: float = 0.5,
        # AdditiveModel params (nested)
        **model_params: Any,
    ) -> None:
        self.min_support = min_support
        self.max_rules = max_rules
        self.PDO = PDO
        self.score0 = score0
        self.odds0 = odds0
        self.bounds_col = bounds_col
        self.on_missing = on_missing
        self.return_sparse = return_sparse
        self.calibrate = calibrate
        self.calibrator__method = calibrator__method
        self.calibrator__n_splits = calibrator__n_splits
        self.calibrator__random_state = calibrator__random_state
        self.calibrator__class_weight = calibrator__class_weight
        self.calibrator__max_iter = calibrator__max_iter
        self.predict_threshold = predict_threshold
        # nested params for AdditiveModel
        self.model_params = dict(model_params)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
            sample_weight: Optional[np.ndarray] = None) -> "RuleCardClassifier":
        start = time.time()
        X_arr = X
        y_arr = np.asarray(y)

        classes = np.unique(y_arr)
        if classes.size != 2:
            raise ValueError("RuleCardClassifier supports binary classification only; "
                             f"got {classes.size} classes {classes}.")
        self.classes_ = classes

        model = AdditiveModel(**self.model_params)
        model.fit(X_arr, y_arr, sample_weight=sample_weight)
        self.additive_model_ = model

        self.scorecard_ = Scorecard.from_model(
            model,
            X_train=X_arr if isinstance(X_arr, pd.DataFrame) else (pd.DataFrame(X_arr)),
            y_train=y_arr,
            min_support=self.min_support,
            max_rules=self.max_rules,
            PDO=self.PDO,
            score0=self.score0,
            odds0=self.odds0,
            bounds_col=self.bounds_col,
            on_missing=self.on_missing,
            return_sparse=self.return_sparse,
        )

        if self.calibrate:
            self.scorecard_.fit_calibrator(
                X_arr if isinstance(X_arr, pd.DataFrame) else (pd.DataFrame(X_arr)),
                y_arr,
                method=self.calibrator__method,
                n_splits=self.calibrator__n_splits,
                random_state=self.calibrator__random_state,
                class_weight=self.calibrator__class_weight,
                max_iter=self.calibrator__max_iter,
            )

        self.feature_names_in_ = getattr(model, "feature_names_in_", None)
        self.n_features_in_ = getattr(model, "n_features_in_", None)
        self.training_info_ = {
            "n_samples": int(len(y_arr)),
            "n_features": int(self.n_features_in_) if self.n_features_in_ is not None else None,
            "class_balance": float(np.mean(y_arr)),
            "fit_time_s": float(time.time() - start),
            "n_rules": int(len(self.scorecard_.rules_df_points)) if hasattr(self.scorecard_, "rules_df_points") else None,
        }
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(self, "scorecard_")
        X_in = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return self.scorecard_.predict_proba(X_in)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        check_is_fitted(self, "scorecard_")
        X_in = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        # use wrapper threshold for convenience; delegate computation to scorecard
        proba = self.scorecard_.predict_proba(X_in)[:, 1]
        return (proba >= float(self.predict_threshold)).astype(int)

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return the additive model margin (log-odds before score scaling)."""
        check_is_fitted(self, "additive_model_")
        return self.additive_model_.decision_function(X)

    # ------------------------------ scorecard helpers ------------------------
    def predict_scores(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Return the **points** score from the scorecard."""
        check_is_fitted(self, "scorecard_")
        X_in = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return self.scorecard_.predict_scores(X_in)

    def rules_table(self) -> pd.DataFrame:
        """Return the rules table with points and metadata."""
        check_is_fitted(self, "scorecard_")
        return self.scorecard_.rules_table()

    # ------------------------------ persistence -----------------------------
    def save(self, path: str) -> None:
        """Persist the trained wrapper (scorecard + additive model)."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "RuleCardClassifier":
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return obj
    # ------------------------------ helpers ---------------------------
    def visualizer(self) -> Any:
        """Return a :class:`ScorecardVisualizer` bound to this model (if available)."""
        check_is_fitted(self, "scorecard_")
        if ScorecardVisualizer is None:
            raise ImportError("ScorecardVisualizer is not available.")
        return ScorecardVisualizer(scorecard=self.scorecard_)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        # top-level params
        params = {
            "min_support": self.min_support,
            "max_rules": self.max_rules,
            "PDO": self.PDO,
            "score0": self.score0,
            "odds0": self.odds0,
            "bounds_col": self.bounds_col,
            "on_missing": self.on_missing,
            "return_sparse": self.return_sparse,
            "calibrate": self.calibrate,
            "calibrator__method": self.calibrator__method,
            "calibrator__n_splits": self.calibrator__n_splits,
            "calibrator__random_state": self.calibrator__random_state,
            "calibrator__class_weight": self.calibrator__class_weight,
            "calibrator__max_iter": self.calibrator__max_iter,
            "predict_threshold": self.predict_threshold,
        }
        # flatten model params with prefix
        for k, v in (self.model_params or {}).items():
            params[f"model__{k}"] = v
        return params

    def set_params(self, **params: Any) -> "RuleCardClassifier":
        for k, v in list(params.items()):
            if k.startswith("model__"):
                key = k.split("__", 1)[1]
                self.model_params[key] = v
                params.pop(k)
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown parameter {k!r} for RuleCardClassifier.")
            setattr(self, k, v)
        return self
