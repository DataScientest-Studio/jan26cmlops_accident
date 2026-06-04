# -*- coding: utf-8 -*-
"""
=============================================================================
  monitoring.py - Monitoring ML avec Evidently
=============================================================================

Compare les données d'entraînement (reference) aux données de prédiction
(current) pour détecter :
  - Data drift : les caractéristiques des accidents ont-elles changé ?
  - Model performance : le modèle se dégrade-t-il ?

Génère des rapports HTML dans reports/.

Prérequis :
  - training_v2.py doit avoir été lancé au moins une fois
    (crée le checkpoint de référence et le modèle)

Commande de lancement :
  python src/models/monitoring.py

Auteur : Projet MLOps Accidents - DataScientest (Marc)
Date   : Mai 2026
=============================================================================
"""

import os
import time
import pandas as pd
import psycopg2
import joblib
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset, ClassificationPreset

# Charger .env
load_dotenv(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".env"
))

# ============================================================================
#  CONFIGURATION
# ============================================================================
# REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = os.getenv("REPO_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")

CONN_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "mlops_accidents"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("POSTGRES_HOST", "db"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
}

MODEL_FILE = os.path.join(MODELS_DIR, "model_focus_tues.pkl")
THRESHOLD_FILE = os.path.join(MODELS_DIR, "threshold_focus_tues.pkl")
FEATURES_FILE = os.path.join(MODELS_DIR, "features_list.pkl")

# Checkpoint d'entraînement = données de référence
# Créé par training_v2.py phase 2 (après nettoyage + feature engineering)
CHECKPOINT_CLEAN = os.path.join(REPO_ROOT, "src", "data", "checkpoint_02_clean.parquet")

# Colonnes à exclure de la comparaison (identifiants, cibles)
COLS_TO_DROP = [
    "num_acc", "num_veh", "id_vehicule",
    "adr", "voie", "v1", "v2",
    "lat", "long",
    "dep", "com",
    "pr", "pr1",
    "grav", "grav_bin",
]


# ============================================================================
#  CHARGEMENT DES DONNÉES
# ============================================================================
def load_reference_data():
    """
    Charge les données d'entraînement sauvegardées lors du dernier training.
    Ce checkpoint sert de référence pour détecter les changements.
    """
    if not os.path.exists(CHECKPOINT_CLEAN):
        raise FileNotFoundError(
            f"Checkpoint introuvable : {CHECKPOINT_CLEAN}\n"
            "Lancez training_v2.py d'abord pour créer les données de référence."
        )
    df = pd.read_parquet(CHECKPOINT_CLEAN)
    print(f"  Reference chargée : {len(df):,} lignes depuis {CHECKPOINT_CLEAN}")
    return df


def load_current_data():
    """
    Charge les données actuelles depuis PostgreSQL.
    Applique la même jointure que training_v2.py phase 1.
    """
    conn = psycopg2.connect(**CONN_PARAMS)

    users = pd.read_sql("SELECT * FROM users;", conn)
    carac = pd.read_sql("SELECT * FROM caracteristics;", conn)
    places = pd.read_sql("SELECT * FROM places;", conn)
    vehicles = pd.read_sql("SELECT * FROM vehicles;", conn)
    conn.close()

    # Normaliser les noms de colonnes
    users.columns = users.columns.str.lower()
    carac.columns = carac.columns.str.lower()
    places.columns = places.columns.str.lower()
    vehicles.columns = vehicles.columns.str.lower()

    # Jointure (même logique que training_v2.py phase 1)
    df = (
        users.merge(vehicles, on=["num_acc", "num_veh"], how="left")
        .merge(carac, on="num_acc", how="left")
        .merge(places, on="num_acc", how="left")
    )
    print(f"  Current chargée : {len(df):,} lignes depuis PostgreSQL")
    return df


# ============================================================================
#  RAPPORT DATA DRIFT
# ============================================================================
def generate_data_drift_report(reference, current):
    """
    Compare la distribution de chaque feature entre reference et current.
    Détecte si les caractéristiques des accidents ont changé.
    """
    print("\n" + "=" * 70)
    print("  RAPPORT DATA DRIFT")
    print("=" * 70)

    # Ne garder que les colonnes numériques communes (hors identifiants)
    ref_num = reference.select_dtypes(include="number")
    cur_num = current.select_dtypes(include="number")
    common = [c for c in ref_num.columns if c in cur_num.columns and c not in COLS_TO_DROP]
    print(f"  Colonnes comparées : {len(common)}")

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=ref_num[common], current_data=cur_num[common])

    output = os.path.join(REPORTS_DIR, "data_drift.html")
    snapshot.save_html(output)
    print(f"  Rapport sauvegardé : {output}")
    return snapshot


# ============================================================================
#  RAPPORT PERFORMANCE MODÈLE
# ============================================================================
def generate_performance_report(reference, current):
    """
    Évalue le modèle sur les données current et compare avec reference.
    Nécessite le modèle entraîné et la colonne cible grav_bin.
    """
    print("\n" + "=" * 70)
    print("  RAPPORT PERFORMANCE MODÈLE")
    print("=" * 70)

    # Vérifier que le modèle existe
    if not os.path.exists(MODEL_FILE):
        print(f"  Modèle introuvable : {MODEL_FILE}")
        print("  Rapport de performance ignoré.")
        return None

    if not os.path.exists(FEATURES_FILE):
        print(f"  Liste de features introuvable : {FEATURES_FILE}")
        print("  Rapport de performance ignoré.")
        return None

    # Charger le modèle et les features
    model = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURES_FILE)

    # Vérifier que grav_bin existe (nécessaire pour évaluer la performance)
    if "grav_bin" not in reference.columns or "grav_bin" not in current.columns:
        print("  Colonne 'grav_bin' absente. Rapport de performance ignoré.")
        return None

    # Prédire sur les deux datasets
    common_feat = [f for f in features if f in reference.columns and f in current.columns]
    print(f"  Features utilisées : {len(common_feat)}")

    ref = reference.copy()
    cur = current.copy()
    ref["prediction"] = model.predict(ref[common_feat])
    cur["prediction"] = model.predict(cur[common_feat])

    report = Report([ClassificationPreset()])
    snapshot = report.run(
        reference_data=ref[common_feat + ["grav_bin", "prediction"]],
        current_data=cur[common_feat + ["grav_bin", "prediction"]],
    )

    output = os.path.join(REPORTS_DIR, "model_performance.html")
    snapshot.save_html(output)
    print(f"  Rapport sauvegardé : {output}")
    return snapshot


# ============================================================================
#  POINT D'ENTRÉE
# ============================================================================
def run_monitoring():
    """
    Lance le monitoring complet.
    Appelé en ligne de commande ou via l'endpoint POST /monitoring/.
    Retourne un dictionnaire avec le statut pour l'API.
    """
    print("=" * 70)
    print("  MONITORING ML - Evidently")
    print("=" * 70)
    t0 = time.time()

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Charger les données
    reference = load_reference_data()
    current = load_current_data()

    # Rapport 1 : Data Drift
    drift_report = generate_data_drift_report(reference, current)

    # Rapport 2 : Performance (si modèle et vérité terrain disponibles)
    perf_report = generate_performance_report(reference, current)

    duree = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  MONITORING TERMINÉ en {duree:.1f}s")
    print(f"  Rapports dans : {REPORTS_DIR}/")
    print("=" * 70)

    return {
        "status": "monitoring completed",
        "duration_seconds": round(duree, 1),
        "data_drift_report": "reports/data_drift.html",
        "performance_report": "reports/model_performance.html" if perf_report else "non généré",
    }


if __name__ == "__main__":
    run_monitoring()
