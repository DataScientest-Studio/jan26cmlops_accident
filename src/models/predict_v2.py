# -*- coding: utf-8 -*-
"""
=============================================================================
  predict_v2.py - Prediction binaire avec seuil optimise focus Tues
=============================================================================

Charge le modele et le seuil produits par training_v2.py, applique le meme
preprocessing que l'entrainement, et predit avec le seuil optimise.

Usage :
  python src/models/predict_v2.py

Auteur : Projet MLOps Accidents - DataScientest (Theodys)
Date   : Avril 2026
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import psycopg2
import joblib
from dotenv import load_dotenv
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# Charger les variables d'environnement depuis .env (racine du projet)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

# ============================================================================
#  CONFIGURATION
# ============================================================================
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "model_focus_tues.pkl")
THRESHOLD_FILE = os.path.join(MODELS_DIR, "threshold_focus_tues.pkl")

# Constantes MLflow
MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")
MODEL_REGISTRY_NAME  = os.getenv("MLFLOW_REGISTRY_NAME", "gravite-focus-tues")

CONN_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "mlops_accidents"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
}

# Colonnes a supprimer (meme liste que training_v2.py)
COLS_TO_DROP = [
    "num_acc", "num_veh", "id_vehicule",
    "adr", "voie", "v1", "v2",
    "lat", "long",
    "dep", "com",
    "pr", "pr1",
    "grav", "grav_bin",
]


# ============================================================================
#  PREPROCESSING (aligne sur training_v2.py)
# ============================================================================
def preprocess_for_predict(df, feature_cols):
    """Applique le meme preprocessing que training_v2.py."""

    # Normaliser colonnes
    df.columns = df.columns.str.lower()

    # GPS 0 -> NaN
    for col_gps in ["lat", "long"]:
        if col_gps in df.columns:
            df[col_gps] = df[col_gps].replace(0, np.nan)

    # "-" -> -1
    for col in df.select_dtypes(include="object").columns:
        df.loc[df[col] == "-", col] = "-1"

    # Feature engineering : age
    if "an_nais" in df.columns and "an" in df.columns:
        df["an_nais"] = pd.to_numeric(df["an_nais"], errors="coerce")
        df["an"] = pd.to_numeric(df["an"], errors="coerce")
        df["age"] = df["an"] - df["an_nais"]
        df.loc[df["age"] < 0, "age"] = np.nan
        df.loc[df["age"] > 120, "age"] = np.nan

    # Feature engineering : heure
    if "hrmn" in df.columns:
        df["hrmn"] = df["hrmn"].astype(str).str.replace(":", "").str.zfill(4)
        df["heure"] = pd.to_numeric(df["hrmn"].str[:2], errors="coerce")

    # Feature engineering : is_weekend
    if "an" in df.columns and "mois" in df.columns and "jour" in df.columns:
        date_tmp = pd.to_datetime(
            df["an"].astype(str) + "-" + df["mois"].astype(str).str.zfill(2) + "-" + df["jour"].astype(str).str.zfill(2),
            errors="coerce",
        )
        df["is_weekend"] = date_tmp.dt.dayofweek.isin([5, 6]).astype(int)

    # Feature engineering : is_holiday (LEFT JOIN holidays)
    try:
        conn = psycopg2.connect(**CONN_PARAMS)
        holidays = pd.read_sql("SELECT * FROM holidays;", conn)
        conn.close()
        holidays.columns = holidays.columns.str.lower()
        if "ds" in holidays.columns and "an" in df.columns and "mois" in df.columns and "jour" in df.columns:
            df["date_acc"] = pd.to_datetime(
                df["an"].astype(str) + "-" + df["mois"].astype(str).str.zfill(2) + "-" + df["jour"].astype(str).str.zfill(2),
                errors="coerce",
            )
            holidays["ds"] = pd.to_datetime(holidays["ds"], errors="coerce")
            holidays["is_holiday"] = 1
            df = df.merge(holidays[["ds", "is_holiday"]], left_on="date_acc", right_on="ds", how="left")
            df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
            df.drop(columns=["ds", "date_acc"], inplace=True, errors="ignore")
        else:
            df["is_holiday"] = 0
    except Exception:
        df["is_holiday"] = 0

    # Encodage object -> numeric
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(-1)
        df[col] = df[col].astype(int)

    df = df.fillna(-1)

    # Selectionner uniquement les features du modele
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  [WARN] Features manquantes : {missing}")
        for c in missing:
            df[c] = -1

    X = df[feature_cols].copy()
    return df, X


# ============================================================================
#  PREDICTION
# ============================================================================
def predict_model():
    """Point d'entree principal - compatible avec api.py."""

    print("\n===== PREDICT_MODEL.PY - Focus Tues =====\n")

    model      = None
    threshold  = 0.5
    feature_cols = None
    model_source = None
    prod_run_id  = None

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        prod_versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["Production"])
        if prod_versions:
            prod_mv     = prod_versions[0]
            prod_run_id = prod_mv.run_id
            model_uri   = f"models:/{MODEL_REGISTRY_NAME}/Production"
            model       = mlflow.xgboost.load_model(model_uri)
            model_source = f"MLflow Registry v{prod_mv.version} (Production)"
            print(f"  Modele charge depuis MLflow : {model_source}")

            # Récupérer le seuil optimal loggué dans le run d'entraînement
            prod_run  = client.get_run(prod_run_id)
            threshold = prod_run.data.metrics.get("threshold", 0.5)
            print(f"  Seuil recupere depuis MLflow : {threshold}")
        else:
            print("  [WARN] Aucun modele en Production dans MLflow")

    except Exception as e:
        print(f"  [WARN] MLflow indisponible ({e})")

    # Fallback : fichier local
    if model is None:
        if not os.path.exists(MODEL_FILE):
            print(f"  [ERREUR] Modele introuvable : {MODEL_FILE}")
            print(f"  Lancez d'abord : python src/models/train_model.py")
            return {"status": "error", "message": "model not found"}
        model        = joblib.load(MODEL_FILE)
        model_source = f"local:{MODEL_FILE}"
        print(f"  Modele charge localement : {MODEL_FILE}")

    # Récupérer les features depuis le fichier threshold (toujours utile)
    if os.path.exists(THRESHOLD_FILE):
        meta         = joblib.load(THRESHOLD_FILE)
        threshold    = meta.get("threshold", threshold)  # MLflow a priorité si déjà défini
        feature_cols = meta["features"]
        print(f"  Features : {len(feature_cols)}")
    else:
        print(f"  [WARN] Seuil introuvable, utilisation du seuil {threshold}")

    # Chargement données PostgreSQL
    conn     = psycopg2.connect(**CONN_PARAMS)
    users    = pd.read_sql("SELECT * FROM users;",         conn)
    carac    = pd.read_sql("SELECT * FROM caracteristics;", conn)
    places   = pd.read_sql("SELECT * FROM places;",        conn)
    vehicles = pd.read_sql("SELECT * FROM vehicles;",      conn)
    conn.close()

    users.columns    = users.columns.str.lower()
    carac.columns    = carac.columns.str.lower()
    places.columns   = places.columns.str.lower()
    vehicles.columns = vehicles.columns.str.lower()

    df = (
        users.merge(vehicles, on=["num_acc", "num_veh"], how="left")
             .merge(carac,    on="num_acc",              how="left")
             .merge(places,   on="num_acc",              how="left")
    )
    df = df.drop_duplicates()
    print(f"  Donnees chargees : {len(df):,} lignes")

    # Preprocessing
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in COLS_TO_DROP]

    df_out, X = preprocess_for_predict(df, feature_cols)

    # Prédiction
    probas      = model.predict_proba(X)[:, 1]
    predictions = (probas >= threshold).astype(int)

    df_out["proba_non_indemne"]       = probas
    df_out["prediction"]        = predictions
    df_out["prediction_label"]  = df_out["prediction"].map({0: "Indemne", 1: "tue/hospit/leger"})

    n_indemne = (predictions == 0).sum()
    n_non_indemne   = (predictions == 1).sum()
    print(f"\n  Predictions :")
    print(f"    Indemne : {n_indemne:,}")
    print(f"    non_indemne   : {n_non_indemne:,}")

    # Focus Tues 
    recall_tues = None
    if "grav" in df_out.columns:
        df_out["grav"] = pd.to_numeric(df_out["grav"], errors="coerce")
        mask_tue = df_out["grav"] == 2
        if mask_tue.sum() > 0:
            tues_detectes = predictions[mask_tue].sum()
            tues_total    = mask_tue.sum()
            recall_tues   = tues_detectes / tues_total
            print(f"\n  Focus Tues :")
            print(f"    Tues dans les donnees  : {tues_total:,}")
            print(f"    Tues detectes (pred=1) : {tues_detectes:,}")
            print(f"    Recall Tues            : {recall_tues:.4f}")

    # Log MLflow du run de prédiction
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT if 'MLFLOW_EXPERIMENT' in dir() else "accidentologie-focus-tues")

        with mlflow.start_run(run_name="prediction") as run:
            mlflow.set_tag("run_type",      "prediction")
            mlflow.set_tag("model_source",  model_source)
            mlflow.set_tag("training_run_id", prod_run_id or "local")

            mlflow.log_param("threshold",      threshold)
            mlflow.log_param("n_features",     len(feature_cols))
            mlflow.log_param("n_predictions",  len(predictions))

            mlflow.log_metric("n_indemne",     int(n_indemne))
            mlflow.log_metric("n_non_indemne",       int(n_non_indemne))
            mlflow.log_metric("ratio_non_indemne",   round(float(n_non_indemne / len(predictions)), 4))

            if recall_tues is not None:
                mlflow.log_metric("recall_tues", round(float(recall_tues), 4))

        print(f"\n  [MLflow] Run prediction loggue : {run.info.run_id}")

    except Exception as e:
        print(f"  [WARN] MLflow prediction log indisponible ({e})")

    # Affichage sample
    print(f"\n  === SAMPLE PREDICTIONS ===")
    cols_display = ["num_acc", "num_veh", "prediction_label", "proba_non_indemne"]
    cols_display = [c for c in cols_display if c in df_out.columns]
    print(df_out[cols_display].head(10).to_string(index=False))

    return {
        "status":           "prediction completed",
        "model_source":     model_source,
        "n_predictions":    len(predictions),
        "n_indemne":        int(n_indemne),
        "n_non_indemne":          int(n_non_indemne),
        "threshold":        float(threshold),
        "recall_tues":      float(recall_tues) if recall_tues is not None else None,
        "sample_predictions": df_out[cols_display].head(10).to_dict(orient="records"),
    }

if __name__ == "__main__":
    result = predict_model()
    print("\n", result)
