from fastapi import FastAPI, HTTPException
from src.models.training_v2 import train_model
from src.models.predict_v2 import predict_model

api = FastAPI(
    title="API - Prédiction de la sévérité d'un accident",
    description="API MLOps - Prediction d'accident",
    version="1.0"
)


@api.post("/training/")
def training_endpoint():
    """Lance l'entraînement du modèle XGBoost."""
    try:
        result = train_model()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/predict/")
def predict_endpoint():
    """Lance les prédictions avec le modèle entraîné."""
    try:
        result = predict_model()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/monitoring/")
def monitoring_endpoint():
    """
    Lance le monitoring Evidently.
    Génère les rapports de data drift et de performance dans reports/.
    Prérequis : training_v2.py doit avoir été lancé au moins une fois.
    """
    try:
        from src.models.monitoring import run_monitoring
        result = run_monitoring()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
