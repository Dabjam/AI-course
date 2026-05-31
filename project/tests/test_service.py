import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import joblib
import pytest
from fastapi.testclient import TestClient

from src.service.app import app, _artifact

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'model.pkl')

SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 840.2,
}


@pytest.fixture(autouse=True)
def load_model():
    _artifact.update(joblib.load(MODEL_PATH))
    yield
    _artifact.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


def test_predict_returns_valid_response(client):
    r = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["churn_prediction"], bool)
    assert data["risk_level"] in ("low", "medium", "high")


def test_predict_low_risk_long_tenure(client):
    customer = {**SAMPLE_CUSTOMER, "tenure": 72, "Contract": "Two year", "MonthlyCharges": 25.0}
    r = client.post("/predict", json=customer)
    assert r.status_code == 200
    assert r.json()["churn_probability"] < 0.5


def test_predict_missing_field(client):
    bad = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "tenure"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
