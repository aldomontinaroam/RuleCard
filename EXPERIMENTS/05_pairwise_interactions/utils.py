import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold
from sklearn.compose import ColumnTransformer

'''
DATASETS:
Adult: https://archive.ics.uci.edu/dataset/2/adult
Breast Cancer: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
COMPAS: https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm
FICO (HELOC): https://www.openml.org/search?type=data&sort=version&status=any&order=asc&exact_name=heloc&id=46932
Mushroom: https://archive.ics.uci.edu/dataset/73/mushroom
Spambase: https://archive.ics.uci.edu/dataset/94/spambase
Titanic: https://www.kaggle.com/competitions/titanic/overview
Telco: https://www.kaggle.com/blastchar/telco-customer-churn/downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv/1
Heart: https://www.kaggle.com/datasets/mragpavank/heart-diseaseuci
Loan: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset
MAGIC gamma: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
German: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data


ref EBM: https://github.com/interpretml/interpret?tab=readme-ov-file
'''

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
    
    # Split X/y
    X = df.drop(columns=class_col)
    y = df[class_col]
    
    # Colonne numeriche e categoriche
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['number']).columns.tolist()

    if one_hot:
        # === One-hot pipeline ===
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

        # Variance threshold
        vt = VarianceThreshold(threshold=var_threshold)
        X_vt = vt.fit_transform(X_df)
        support_mask = vt.get_support()
        selected_features = [f for f, keep in zip(feature_names, support_mask) if keep]
        X_df = pd.DataFrame(X_vt, columns=selected_features)
        feature_names = selected_features

        # Scaling
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_df), columns=selected_features)

    else:
        # === Native categorical processing ===
        num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])
        cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))])

        # Imputazione
        if num_cols:
            X[num_cols] = num_pipeline.fit_transform(X[num_cols])
        if cat_cols:
            X[cat_cols] = cat_pipeline.fit_transform(X[cat_cols])

        # Converti in categorico esplicito
        for col in cat_cols:
            X[col] = X[col].astype('category')

        # Scaling solo delle numeriche
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
