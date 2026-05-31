import os
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.data.preprocess import load_raw, clean, build_encoders, encode, split, get_feature_names

DATA_PATH = os.path.join("data", "telco_churn.csv")
ARTIFACT_PATH = os.path.join("artifacts", "model.pkl")


def get_metrics(model, X, y, name):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "model": name,
        "ROC-AUC": round(roc_auc_score(y, y_proba), 4),
        "F1": round(f1_score(y, y_pred), 4),
        "Precision": round(precision_score(y, y_pred), 4),
        "Recall": round(recall_score(y, y_pred), 4),
    }


def main():
    df = load_raw(DATA_PATH)
    df = clean(df)
    encoders = build_encoders(df)
    df = encode(df, encoders)

    X_train, X_val, X_test, y_train, y_val, y_test = split(df)
    features = get_feature_names()
    X_train = X_train[features]
    X_val = X_val[features]
    X_test = X_test[features]

    models = {
        "Dummy": DummyClassifier(strategy="stratified", random_state=42),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    results = []
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        results.append(get_metrics(model, X_val, y_val, name))

    df_res = pd.DataFrame(results).set_index("model")
    print("\nРезультаты на val:")
    print(df_res.to_string())

    best_name = df_res["ROC-AUC"].idxmax()
    best_model = trained[best_name]
    print(f"\nЛучшая модель: {best_name}")

    test_m = get_metrics(best_model, X_test, y_test, best_name)
    print(f"\nМетрики на тесте ({best_name}):")
    for k, v in test_m.items():
        if k != "model":
            print(f"  {k}: {v}")

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump({"model": best_model, "features": features, "encoders": encoders}, ARTIFACT_PATH)
    print(f"\nМодель сохранена в {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
