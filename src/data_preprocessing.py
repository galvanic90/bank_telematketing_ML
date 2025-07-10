from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler



def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()
    return df

def count_yes_no(df):
    number_of_values = df.y.value_counts()
    total_values =  number_of_values['no'] + number_of_values['yes']
    yes_percentage = number_of_values['yes'] / total_values
    no_percentage = number_of_values['no'] / total_values
    print(number_of_values)
    print(yes_percentage)
    print(no_percentage)
    
    

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
