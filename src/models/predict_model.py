# -*- coding: utf-8 -*-
"""
=============================================================================
  predict_model.py - Prediction binaire avec seuil optimise focus Tues
=============================================================================

Charge le modele et le seuil produits par train_model.py, applique le meme
preprocessing que l'entrainement, et predit avec le seuil optimise.

Usage :
  python src/models/predict_model.py

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

# Charger les variables d'environnement depuis .env (racine du projet)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

# ============================================================================
#  CONFIGURATION
# ============================================================================
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "model_focus_tues.pkl")
THRESHOLD_FILE = os.path.join(MODELS_DIR, "threshold_focus_tues.pkl")

CONN_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "mlops_accidents"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
}

# Colonnes a supprimer (meme liste que train_model.py)
COLS_TO_DROP = [
    "num_acc", "num_veh", "id_vehicule",
    "adr", "voie", "v1", "v2",
    "lat", "long",
    "dep", "com",
    "pr", "pr1",
    "grav", "grav_bin",
]


# ============================================================================
#  PREPROCESSING (aligne sur train_model.py)
# ============================================================================
def preprocess_for_predict(df, feature_cols):
    """Applique le meme preprocessing que train_model.py."""

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

    # Charger modele et seuil
    if not os.path.exists(MODEL_FILE):
        print(f"  [ERREUR] Modele introuvable : {MODEL_FILE}")
        print(f"  Lancez d'abord : python src/models/train_model.py")
        return {"status": "error", "message": "model not found"}

    model = joblib.load(MODEL_FILE)
    print(f"  Modele charge : {MODEL_FILE}")

    if os.path.exists(THRESHOLD_FILE):
        meta = joblib.load(THRESHOLD_FILE)
        threshold = meta["threshold"]
        feature_cols = meta["features"]
        print(f"  Seuil : {threshold}")
        print(f"  Features : {len(feature_cols)}")
    else:
        print(f"  [WARN] Seuil introuvable, utilisation du seuil 0.5")
        threshold = 0.5
        feature_cols = None

    # Charger les donnees depuis PostgreSQL
    conn = psycopg2.connect(**CONN_PARAMS)
    users = pd.read_sql("SELECT * FROM users;", conn)
    carac = pd.read_sql("SELECT * FROM caracteristics;", conn)
    places = pd.read_sql("SELECT * FROM places;", conn)
    vehicles = pd.read_sql("SELECT * FROM vehicles;", conn)
    conn.close()

    # Normaliser
    users.columns = users.columns.str.lower()
    carac.columns = carac.columns.str.lower()
    places.columns = places.columns.str.lower()
    vehicles.columns = vehicles.columns.str.lower()

    # Jointure
    df = (
        users.merge(vehicles, on=["num_acc", "num_veh"], how="left")
        .merge(carac, on="num_acc", how="left")
        .merge(places, on="num_acc", how="left")
    )
    df = df.drop_duplicates()

    print(f"  Donnees chargees : {len(df):,} lignes")

    # Preprocessing
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in COLS_TO_DROP]

    df_out, X = preprocess_for_predict(df, feature_cols)

    # Prediction avec seuil optimise
    probas = model.predict_proba(X)[:, 1]
    predictions = (probas >= threshold).astype(int)

    df_out["proba_reste"] = probas
    df_out["prediction"] = predictions
    df_out["prediction_label"] = df_out["prediction"].map({0: "Indemne", 1: "Reste (tue/hospit/leger)"})

    # Stats
    n_indemne = (predictions == 0).sum()
    n_reste = (predictions == 1).sum()
    print(f"\n  Predictions :")
    print(f"    Indemne : {n_indemne:,}")
    print(f"    Reste   : {n_reste:,}")

    # Comparaison avec grav reel si disponible
    if "grav" in df_out.columns:
        df_out["grav"] = pd.to_numeric(df_out["grav"], errors="coerce")
        mask_tue = df_out["grav"] == 2
        if mask_tue.sum() > 0:
            tues_detectes = predictions[mask_tue].sum()
            tues_total = mask_tue.sum()
            print(f"\n  Focus Tues :")
            print(f"    Tues dans les donnees : {tues_total:,}")
            print(f"    Tues detectes (pred=1) : {tues_detectes:,}")
            print(f"    Recall Tues : {tues_detectes / tues_total:.4f}")

    print(f"\n  === SAMPLE PREDICTIONS ===")
    cols_display = ["num_acc", "num_veh", "prediction_label", "proba_reste"]
    cols_display = [c for c in cols_display if c in df_out.columns]
    print(df_out[cols_display].head(10).to_string(index=False))

    return {
        "status": "prediction completed",
        "n_predictions": len(predictions),
        "n_indemne": int(n_indemne),
        "n_reste": int(n_reste),
        "threshold": float(threshold),
        "sample_predictions": df_out[cols_display].head(10).to_dict(orient="records"),
    }


if __name__ == "__main__":
    result = predict_model()
    print("\n", result)
