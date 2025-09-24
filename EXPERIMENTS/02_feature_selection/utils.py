import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd

def get_available_datasets():
    return {
        'adult': {
            'path': 'DATA/adult.csv',
            'class_col': 'class',
            'positive_class': '>50K',
            'idCol': None
        },
        'breast_cancer': {
            'path': 'DATA/breast_cancer.csv',
            'class_col': 'target',
            'positive_class': '1',
            'idCol': None
        },
        'compas': {
            'path': 'DATA/compas-scores-two-years.csv',
            'class_col': 'two_year_recid',
            'positive_class': '1',
            'idCol': 'id'
        },
        'fico': {
            'path': 'DATA/fico.csv',
            'class_col': 'RiskPerformance',
            'positive_class': 'Good',
            'idCol': None
        },
        'mushroom': {
            'path': 'DATA/mushroom.csv',
            'class_col': 'class',
            'positive_class': 'e',
            'idCol': None
        },
        'spambase': {
            'path': 'DATA/spambase.csv',
            'class_col': 'spam',
            'positive_class': '1',
            'idCol': None
        },
        'titanic': {
            'path': 'DATA/titanic.csv',
            'class_col': 'Survived',
            'positive_class': '1',
            'idCol': 'PassengerId'
        },
        'telco': {
            'path': 'DATA/telco.csv',
            'class_col': 'Churn',
            'positive_class': 'Yes',
            'idCol': 'customerID'
        },
        'heart': {
            'path': 'DATA/heart.csv',
            'class_col': 'target',
            'positive_class': '1',
            'idCol':None
        },
        'loan': {
            'path': 'DATA/loan.csv',
            'class_col': 'loan_status',
            'positive_class': 'Approved',
            'idCol': 'loan_id'
        },
        'magic_gamma': {
            'path': 'DATA/magic_gamma.csv',
            'class_col': 'class',
            'positive_class': 'g',
            'idCol': None
        },
        'german': {
            'path': 'DATA/german_credit.csv',
            'class_col': 'default',
            'positive_class': '1',
            'idCol':None
        }
    }

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

def load_preprocess(name, one_hot=True, var_threshold=0.001):
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
    
    print(f"Dataset: {name}")
    print(f"Target: {mapping}")
    print(f"Shape: {df.shape}")

    df[class_col] = df[class_col].map(mapping)
    
    X = df.drop(columns=class_col)
    y = df[class_col]
    
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['number']).columns.tolist()

    if one_hot:
        num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32))
        ])

        transformers = []
        if num_cols:
            transformers.append(('num', num_pipeline, num_cols))
        if cat_cols:
            transformers.append(('cat', cat_pipeline, cat_cols))
            
        preprocessor = ColumnTransformer(transformers)
        X_enc = preprocessor.fit_transform(X)
        
        feat_names = preprocessor.get_feature_names_out()
        feature_names = list(dict.fromkeys(_clean_column_names(feat_names)))

        X_df = pd.DataFrame(X_enc, columns=feature_names)

        vt = VarianceThreshold(threshold=var_threshold)
        X_vt = vt.fit_transform(X_df)
        support_mask = vt.get_support()
        selected_features = [f for f, keep in zip(feature_names, support_mask) if keep]
        X_df = pd.DataFrame(X_vt, columns=selected_features)
        feature_names = selected_features

        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_df), columns=selected_features)

    else:
        num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])
        cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))])

        if num_cols:
            X[num_cols] = num_pipeline.fit_transform(X[num_cols])
        if cat_cols:
            X[cat_cols] = cat_pipeline.fit_transform(X[cat_cols])

        for col in cat_cols:
            X[col] = X[col].astype('category')

        scaler = MinMaxScaler()
        if num_cols:
            X[num_cols] = scaler.fit_transform(X[num_cols])

        X_scaled = X.copy()
        feature_names = num_cols + cat_cols
        preprocessor = None 

    print(f"Features: {len(feature_names)}")

    return {
        "data": pd.concat([X_scaled, y], axis=1),
        "X": X_scaled,
        "y": y,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "scaler": scaler,
        "shape": X_scaled.shape
    }

methods_need_scaling = {
        "logistic_regression_l2",
        "logistic_regression_l1",
        "rfe_logistic_l2",
        "rfe_logistic_l1",
        "select_kbest_chi2",
        "select_kbest"
    }


def analyze_method_consistency(scores_per_dataset):
    summary_list = []

    for entry in scores_per_dataset:
        dataset_name = entry["dataset"]
        df = entry["scores_df"]
        for _, row in df.iterrows():
            summary_list.append({
                "dataset": dataset_name,
                "method": row["method"],
                "SCORE": row["SCORE"]
            })

    summary_df = pd.DataFrame(summary_list)

    summary_df["rank"] = summary_df.groupby("dataset")["SCORE"].rank(ascending=False, method="min")

    var_df = summary_df.groupby("method")["rank"].var().reset_index()
    var_df.columns = ["method", "rank_variance"]

    std_df = summary_df.groupby("method")["SCORE"].std().reset_index()
    std_df.columns = ["method", "score_std"]

    range_df = summary_df.groupby("method").agg(score_max=("SCORE", "max"),
                                                score_min=("SCORE", "min")).reset_index()
    range_df["score_range"] = range_df["score_max"] - range_df["score_min"]

    mean_df = summary_df.groupby("method")["SCORE"].mean().reset_index()
    mean_df.columns = ["method", "mean_SCORE"]

    final_df = var_df.merge(std_df, on="method").merge(range_df[["method", "score_range"]], on="method").merge(mean_df, on="method")

    final_df["needs_standardization"] = final_df["method"].apply(lambda x: 1 if x in methods_need_scaling else 0)

    epsilon = 1e-6

    w1 = 0.2
    w2 = 0.2
    w3 = 0.2
    w4 = 0.4

    final_df['SCORE_TOTAL'] = (
        w1 * 1 / (final_df['rank_variance'] + epsilon) +
        w2 * 1 / (final_df['score_std'] + epsilon) +
        w3 * 1 / (final_df['score_range'] + epsilon) +
        w4 * final_df['mean_SCORE'] - final_df['needs_standardization'] * 0.1
    )

    min_score = final_df['SCORE_TOTAL'].min()
    max_score = final_df['SCORE_TOTAL'].max()
    final_df['SCORE_TOTAL'] = (final_df['SCORE_TOTAL'] - min_score) / (max_score - min_score)



    final_df = final_df.sort_values("SCORE_TOTAL", ascending=False).reset_index(drop=True)
    final_df = final_df.round({
        "rank_variance": 2,
        "score_std": 2,
        "score_range": 2,
        "mean_SCORE": 2
    })

    return final_df
