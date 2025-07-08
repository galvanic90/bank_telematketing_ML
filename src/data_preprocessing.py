from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler



def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    return df



def encode_and_scale(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if 'y' not in df.columns:
        raise ValueError(f"'y' not found in columns: {df.columns.tolist()}")

    y = df['y'].map({'yes': 1, 'no': 0})
    X = df.drop('y', axis=1)

    categorical = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical, drop_first=True)

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])

    return X_encoded, y
