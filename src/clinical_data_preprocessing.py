import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from src import config

def preprocess_clinical_data():
    df = pd.read_excel(config.DATA_PATH)
    print(f" Initial: {len(df)} patients")

    # Pulizia e indixization using US_Number
    df = df[df['US_Number'].notna()].copy()
    df['US_Number'] = df['US_Number'].astype(str).str.extract(r'^(\d+)')[0].str.lstrip('0')
    df = df.drop_duplicates(subset='US_Number').set_index('US_Number')
    print(f" After cleaning: {len(df)} patients")

    #Map target labels
    df = df[df['Diagnosis'].isin(['appendicitis', 'no appendicitis'])]
    y = df['Diagnosis'].map({'appendicitis': 1, 'no appendicitis': 0})

    drop_cols = ['Diagnosis', 'Management', 'Severity', 'Diagnosis_Presumptive']
    X = df.drop(columns = drop_cols, errors='ignore')
    print(f" Rows with valid target: {len(X)}")

    # drop columns with > 40% NaN
    thresh = 0.4
    X = X.loc[:, X.isna().mean() <= thresh]
    print(f" Columns after Nan filtering: {X.shape[1]}")

    # Imputation missing remaining values
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    X = pd.get_dummies(X, drop_first=True)

    # Remove low variance features
    selector = VarianceThreshold(threshold=0.01)
    X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()], index=X.index)

    # Robust Scaling
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    print(f" Output: {X_scaled.shape[0]} patients, {X_scaled.shape[1]} feature")
    print(f" Target distribution:")
    print(y.value_counts(normalize=True))

    return X_scaled,y
