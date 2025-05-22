import pandas as pd
from sklearn.preprocessing import StandardScaler
from src import config

def preprocess_clinical_data():
    # caricamento del dataset
    df = pd.read_excel(config.DATA_PATH)

    # selezione delle feature e del target
    X = df.drop(columns=['Diagnosis', 'Management', 'Severity'])
    y = df['Diagnosis'].map({'appendicitis': 1, 'no appendicitis': 0})

    # rimozione delle righe con target mancante
    y = y.dropna()
    X = X.loc[y.index]  # Mantiene solo le righe corrispondenti a y_diagnosis non-NaN

    # rimozione delle colonne con troppi valori mancanti
    X = X.dropna(axis=1, thresh=0.4 * len(X))  # 40% dei valori mancanti

    # Imputazione dei valori mancanti rimanenti
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # Codifica delle variabili categoriali
    X = pd.get_dummies(X, drop_first=True)

    # Standardizzazione delle variabili numeriche
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X,y
