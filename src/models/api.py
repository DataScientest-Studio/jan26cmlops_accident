from fastapi import FastAPI, HTTPException
from training import train_model
from predict import predict_model

api = FastAPI(
    title="API - Prédiction de la sévérité d'un accidents",
    description="API MLOps - Prediction d'accident",
    version="1.0"
)


@api.post("/training/")
def training_endpoint():
    try:
        result = train_model()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/predict/")
def predict_endpoint():
    try:
        result = predict_model()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))