import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

TARGET_COL = "Churn"
DROP_COLS = ["customerID"]


def load_raw(path):
    df = pd.read_csv(path)
    return df


def clean(df):
    df = df.copy()
    df = df.drop(columns=DROP_COLS)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)
    return df


def build_encoders(df):
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


def encode(df, encoders):
    df = df.copy()
    for col in CATEGORICAL_COLS:
        df[col] = encoders[col].transform(df[col].astype(str))
    return df


def split(df, test_size=0.2, val_size=0.1, random_state=42):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=random_state, stratify=y_train
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_feature_names():
    return CATEGORICAL_COLS + NUMERIC_COLS


def prepare_single(data, encoders):
    df = pd.DataFrame([data])
    df = encode(df, encoders)
    return df[get_feature_names()]
