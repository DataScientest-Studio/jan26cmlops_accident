import os
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import mlflow
from mlflow.tracking import MlflowClient

from src.models.training_v2 import (
    train_model,
    CONN_PARAMS,
    REPORTS_DIR,
    FIGURES_DIR,
    REPORT_FILE,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    MODEL_REGISTRY_NAME,
)
from src.models.predict_v2 import predict_model

api = FastAPI(
    title="API - Prédiction de la sévérité d'un accident",
    description="API MLOps - Prediction d'accident",
    version="1.0"
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mlflow_client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()

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
    """Lance le monitoring Evidently (data drift + performance)."""
    try:
        from src.models.monitoring import run_monitoring
        result = run_monitoring()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
#  HEALTH CHECK
# ============================================================================
@api.get("/health")
def health():
    """Vérification rapide que l'API répond."""
    return {"status": "ok"}


# ============================================================================
#  STATS SUR LES DONNÉES (Streamlit - chapitre Scripts ML)
# ============================================================================
@api.get("/data/stats")
def data_stats():
    """Statistiques générales sur le dataset accidentologie."""
    try:
        conn = psycopg2.connect(**CONN_PARAMS)
        cur = conn.cursor()

        # Compte par table
        table_counts = {}
        for table in ["users", "vehicles", "caracteristics", "places"]:
            cur.execute(f"SELECT COUNT(*) FROM {table};")
            table_counts[table] = cur.fetchone()[0]

        # Distribution de la gravité
        cur.execute("SELECT grav, COUNT(*) FROM users GROUP BY grav ORDER BY grav;")
        grav_distribution = {str(row[0]): row[1] for row in cur.fetchall()}

        # Plage d'années
        cur.execute("SELECT MIN(an), MAX(an) FROM caracteristics;")
        an_min, an_max = cur.fetchone()

        # Nombre d'accidents par année
        cur.execute("SELECT an, COUNT(DISTINCT num_acc) FROM caracteristics GROUP BY an ORDER BY an;")
        accidents_per_year = {str(row[0]): row[1] for row in cur.fetchall()}

        conn.close()

        return {
            "table_counts": table_counts,
            "grav_distribution": grav_distribution,
            "grav_labels": {
                "1": "Indemne",
                "2": "Tué",
                "3": "Hospitalisé",
                "4": "Blessé léger",
            },
            "year_range": {"min": an_min, "max": an_max},
            "accidents_per_year": accidents_per_year,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
#  MLFLOW - RUNS & MODEL REGISTRY (Streamlit - chapitre Suivi MLflow)
# ============================================================================
@api.get("/mlflow/runs")
def mlflow_runs(experiment_name: str = MLFLOW_EXPERIMENT, max_results: int = 30):
    """Liste les derniers runs MLflow avec leurs métriques et params."""
    try:
        client = _mlflow_client()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return {"runs": []}

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=max_results,
        )

        out = []
        for r in runs:
            out.append({
                "run_id":     r.info.run_id,
                "run_name":   r.data.tags.get("mlflow.runName", ""),
                "run_type":   r.data.tags.get("run_type", "training"),
                "status":     r.info.status,
                "start_time": r.info.start_time,
                "metrics":    r.data.metrics,
                "params": {
                    k: v for k, v in r.data.params.items()
                    if not k.startswith("dvc_hash")
                },
            })
        return {"runs": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/mlflow/model")
def mlflow_model_info():
    """Informations sur le modèle actuellement en Production."""
    try:
        client = _mlflow_client()
        versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["Production"])
        if not versions:
            return {"status": "no production model"}

        mv  = versions[0]
        run = client.get_run(mv.run_id)

        return {
            "name":    MODEL_REGISTRY_NAME,
            "version": mv.version,
            "run_id":  mv.run_id,
            "stage":   mv.current_stage,
            "metrics": run.data.metrics,
            "params": {
                k: v for k, v in run.data.params.items()
                if not k.startswith("dvc_hash")
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
#  RAPPORTS & FIGURES (Streamlit - chapitres Scripts ML, Monitoring, Démo)
# ============================================================================
@api.get("/reports/figures/{filename}")
def get_figure(filename: str):
    """Sert une image PNG (ROC, PR, Feature Importance, Confusion Matrix)."""
    path = os.path.join(FIGURES_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Figure introuvable : {filename}")
    return FileResponse(path, media_type="image/png")


@api.get("/reports/html/{filename}")
def get_html_report(filename: str):
    """Sert un rapport HTML Evidently (data_drift.html, model_performance.html)."""
    path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Rapport introuvable : {filename}")
    return FileResponse(path, media_type="text/html")


@api.get("/reports/training-text")
def get_training_report_text():
    """Retourne le rapport texte de training (toutes les phases)."""
    if not os.path.exists(REPORT_FILE):
        raise HTTPException(status_code=404, detail="Rapport de training introuvable")
    with open(REPORT_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    return {"content": content}