import logging
import os
import time
from contextlib import asynccontextmanager

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data.preprocess import prepare_single

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("artifacts", "model.pkl"))

_artifact = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Загружаем модель из %s", MODEL_PATH)
    _artifact.update(joblib.load(MODEL_PATH))
    logger.info("Модель загружена")
    yield
    _artifact.clear()


app = FastAPI(title="Churn Prediction", version="1.0.0", lifespan=lifespan)


class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _artifact}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if "model" not in _artifact:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    t0 = time.time()
    try:
        X = prepare_single(customer.model_dump(), _artifact["encoders"])
        proba = float(_artifact["model"].predict_proba(X)[0, 1])
    except Exception as e:
        logger.error("Ошибка предсказания: %s", e)
        raise HTTPException(status_code=422, detail=str(e))

    elapsed = round((time.time() - t0) * 1000, 1)

    if proba >= 0.7:
        risk = "high"
    elif proba >= 0.4:
        risk = "medium"
    else:
        risk = "low"

    logger.info("predict | proba=%.3f risk=%s time_ms=%s", proba, risk, elapsed)

    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=proba >= 0.5,
        risk_level=risk,
    )
