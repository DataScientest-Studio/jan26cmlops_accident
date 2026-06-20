# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import psycopg2
import joblib
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset, ClassificationPreset

load_dotenv(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".env"
))

# ============================================================================
#  CONFIGURATION
# ============================================================================
REPO_ROOT   = os.getenv("REPO_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODELS_DIR  = os.path.join(REPO_ROOT, "models")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")

CONN_PARAMS = {
    "dbname":   os.getenv("POSTGRES_DB",      "mlops_accidents"),
    "user":     os.getenv("POSTGRES_USER",     "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host":     os.getenv("POSTGRES_HOST",     "db"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
}

MODEL_FILE      = os.path.join(MODELS_DIR, "model_focus_tues.pkl")
THRESHOLD_FILE  = os.path.join(MODELS_DIR, "threshold_focus_tues.pkl")
FEATURES_FILE   = os.path.join(MODELS_DIR, "features_list.pkl")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:8080")
MLFLOW_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT",   "accidentologie-focus-tues")

# Année de coupure : historique < SPLIT_YEAR <= récent
SPLIT_YEAR = int(os.getenv("MONITORING_SPLIT_YEAR", "16"))

# Seuil de drift pour déclencher le retraining
DRIFT_SHARE_THRESHOLD = float(os.getenv("DRIFT_SHARE_THRESHOLD", "0.3"))

MAX_ROWS = int(os.getenv("MONITORING_MAX_ROWS", "10000"))

COLS_TO_DROP = [
    "num_acc", "num_veh", "id_vehicule",
    "adr", "voie", "v1", "v2",
    "lat", "long", "dep", "com",
    "pr", "pr1", "grav", "grav_bin",
]


# ============================================================================
#  CHARGEMENT ET PREPROCESSING
# ============================================================================
def _load_and_join() -> pd.DataFrame:
    """Charge et joint les 4 tables depuis PostgreSQL."""
    conn     = psycopg2.connect(**CONN_PARAMS)
    users    = pd.read_sql("SELECT * FROM users;",          conn)
    carac    = pd.read_sql("SELECT * FROM caracteristics;", conn)
    places   = pd.read_sql("SELECT * FROM places;",         conn)
    vehicles = pd.read_sql("SELECT * FROM vehicles;",       conn)
    conn.close()

    for df in [users, carac, places, vehicles]:
        df.columns = df.columns.str.lower()

    return (
        users.merge(vehicles, on=["num_acc", "num_veh"], how="left")
             .merge(carac,    on="num_acc",              how="left")
             .merge(places,   on="num_acc",              how="left")
             .drop_duplicates()
    )


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Applique le même preprocessing que training_v2.py phase 2."""

    # GPS 0 -> NaN
    for col in ["lat", "long"]:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # "-" -> -1
    for col in df.select_dtypes(include="object").columns:
        df.loc[df[col] == "-", col] = "-1"

    # Feature engineering
    if "an_nais" in df.columns and "an" in df.columns:
        df["an_nais"] = pd.to_numeric(df["an_nais"], errors="coerce")
        df["an"]      = pd.to_numeric(df["an"],      errors="coerce")
        df["age"]     = df["an"] - df["an_nais"]
        df.loc[df["age"] < 0,   "age"] = np.nan
        df.loc[df["age"] > 120, "age"] = np.nan

    if "hrmn" in df.columns:
        df["hrmn"]  = df["hrmn"].astype(str).str.replace(":", "").str.zfill(4)
        df["heure"] = pd.to_numeric(df["hrmn"].str[:2], errors="coerce")

    if all(c in df.columns for c in ["an", "mois", "jour"]):
        date_tmp = pd.to_datetime(
            df["an"].astype(str) + "-" +
            df["mois"].astype(str).str.zfill(2) + "-" +
            df["jour"].astype(str).str.zfill(2),
            errors="coerce",
        )
        df["is_weekend"] = date_tmp.dt.dayofweek.isin([5, 6]).astype(int)

    df["is_holiday"] = 0

    # Cible binaire
    df["grav"]     = pd.to_numeric(df["grav"], errors="coerce")
    df             = df.dropna(subset=["grav"])
    df["grav"]     = df["grav"].astype(int)
    df["grav_bin"] = (df["grav"] != 1).astype(int)

    # Encodage object -> numeric
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)

    return df.fillna(-1)


def load_reference_and_current() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge toutes les données, préprocesse, puis sépare :
      - référence  : années < SPLIT_YEAR  (historique)
      - current    : années >= SPLIT_YEAR (récent)
    """
    print(f"  Chargement PostgreSQL...")
    df = _load_and_join()
    print(f"  Jointure : {len(df):,} lignes")

    df = _preprocess(df)
    print(f"  Après preprocessing : {len(df):,} lignes")

    # Séparation par année
    if "an" not in df.columns:
        raise ValueError("Colonne 'an' introuvable — impossible de séparer historique/récent.")

    df["an"] = pd.to_numeric(df["an"], errors="coerce")
    reference = df[df["an"] <  SPLIT_YEAR].copy()
    current   = df[df["an"] >= SPLIT_YEAR].copy()

    print(f"  Référence (an < {SPLIT_YEAR})  : {len(reference):,} lignes")
    print(f"  Current   (an >= {SPLIT_YEAR}) : {len(current):,} lignes")

    if len(reference) == 0:
        raise ValueError(f"Aucune donnée historique (an < {SPLIT_YEAR})")
    if len(current) == 0:
        raise ValueError(f"Aucune donnée récente (an >= {SPLIT_YEAR})")

    return reference, current


# ============================================================================
#  RAPPORT DATA DRIFT
# ============================================================================
def generate_data_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("  RAPPORT DATA DRIFT")
    print("=" * 70)

    ref_num = reference.select_dtypes(include="number")
    cur_num = current.select_dtypes(include="number")
    common  = [c for c in ref_num.columns if c in cur_num.columns and c not in COLS_TO_DROP]
    print(f"  Colonnes comparées : {len(common)}")

    # Sous-échantillonnage
    ref_s = ref_num[common].sample(n=min(MAX_ROWS, len(ref_num)), random_state=42)
    cur_s = cur_num[common].sample(n=min(MAX_ROWS, len(cur_num)), random_state=42)

    report   = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=ref_s, current_data=cur_s)

    # Sauvegarde HTML
    output = os.path.join(REPORTS_DIR, "data_drift.html")
    snapshot.save_html(output)
    print(f"  Rapport HTML : {output}")

    # Extraction des métriques de drift
    result_dict  = snapshot.dict()
    metrics_raw  = result_dict.get("metrics", [])
    drift_share  = 0.0
    n_drifted    = 0
    n_total      = 0

    for m in metrics_raw:
        result = m.get("result", {})
        if "share_of_drifted_columns" in result:
            drift_share = result["share_of_drifted_columns"]
        if "number_of_drifted_columns" in result:
            n_drifted = result["number_of_drifted_columns"]
        if "number_of_columns" in result:
            n_total = result["number_of_columns"]

    print(f"  Colonnes driftées : {n_drifted}/{n_total} ({drift_share:.1%})")

    return {
        "drift_share":    drift_share,
        "n_drifted":      n_drifted,
        "n_total":        n_total,
        "report_path":    output,
        "snapshot":       snapshot,
    }


# ============================================================================
#  RAPPORT PERFORMANCE MODÈLE
# ============================================================================
def generate_performance_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("  RAPPORT PERFORMANCE MODÈLE")
    print("=" * 70)

    if not os.path.exists(MODEL_FILE) or not os.path.exists(FEATURES_FILE):
        print("  Modèle ou features introuvables — rapport ignoré.")
        return {}

    model    = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURES_FILE)

    common_feat = [f for f in features if f in reference.columns and f in current.columns]
    print(f"  Features utilisées : {len(common_feat)}")

    if "grav_bin" not in reference.columns or "grav_bin" not in current.columns:
        print("  Colonne 'grav_bin' absente — rapport ignoré.")
        return {}

    # Sous-échantillonnage
    ref_s = reference.sample(n=min(MAX_ROWS, len(reference)), random_state=42).copy()
    cur_s = current.sample(n=min(MAX_ROWS, len(current)),    random_state=42).copy()

    ref_s["prediction"] = model.predict(ref_s[common_feat])
    cur_s["prediction"] = model.predict(cur_s[common_feat])

    report   = Report([ClassificationPreset()])
    snapshot = report.run(
        reference_data=ref_s[common_feat + ["grav_bin", "prediction"]],
        current_data=cur_s[common_feat   + ["grav_bin", "prediction"]],
    )

    output = os.path.join(REPORTS_DIR, "model_performance.html")
    snapshot.save_html(output)
    print(f"  Rapport HTML : {output}")

    # Extraction métriques
    result_dict = snapshot.dict()
    f1_ref, f1_cur = 0.0, 0.0
    for m in result_dict.get("metrics", []):
        result = m.get("result", {})
        if "reference" in result and "current" in result:
            f1_ref = result["reference"].get("f1",  f1_ref)
            f1_cur = result["current"].get("f1",    f1_cur)

    print(f"  F1 référence : {f1_ref:.4f} | F1 current : {f1_cur:.4f}")

    return {
        "f1_reference": f1_ref,
        "f1_current":   f1_cur,
        "report_path":  output,
        "snapshot":     snapshot,
    }


# ============================================================================
#  LOG MLFLOW
# ============================================================================
def log_to_mlflow(drift_results: dict, perf_results: dict, retrain_triggered: bool) -> str:
    """Log les métriques et artefacts Evidently dans MLflow."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run(run_name="monitoring") as run:
            mlflow.set_tag("run_type", "monitoring")
            mlflow.set_tag("split_year", str(SPLIT_YEAR))

            # Métriques drift
            if drift_results:
                mlflow.log_metric("drift_share",   drift_results.get("drift_share",  0.0))
                mlflow.log_metric("n_drifted",     drift_results.get("n_drifted",    0))
                mlflow.log_metric("n_total_cols",  drift_results.get("n_total",      0))
                mlflow.log_param("drift_threshold", DRIFT_SHARE_THRESHOLD)

            # Métriques performance
            if perf_results:
                mlflow.log_metric("f1_reference", perf_results.get("f1_reference", 0.0))
                mlflow.log_metric("f1_current",   perf_results.get("f1_current",   0.0))

            # Tag retraining
            mlflow.set_tag("retrain_triggered", str(retrain_triggered))

            # Artefacts HTML
            if drift_results.get("report_path") and os.path.exists(drift_results["report_path"]):
                mlflow.log_artifact(drift_results["report_path"])
            if perf_results.get("report_path") and os.path.exists(perf_results["report_path"]):
                mlflow.log_artifact(perf_results["report_path"])

            run_id = run.info.run_id
            print(f"\n  [MLflow] Run monitoring loggué : {run_id}")
            return run_id

    except Exception as e:
        print(f"  [WARN] MLflow indisponible : {e}")
        return ""


# ============================================================================
#  POINT D'ENTRÉE
# ============================================================================
def run_monitoring():
    print("=" * 70)
    print("  MONITORING ML - Evidently")
    print("=" * 70)
    t0 = time.time()

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Chargement et séparation historique / récent
    reference, current = load_reference_and_current()

    # 2. Rapport Data Drift
    drift_results = generate_data_drift_report(reference, current)

    # 3. Rapport Performance
    perf_results = generate_performance_report(reference, current)

    # 4. Décision retraining
    drift_share       = drift_results.get("drift_share", 0.0)
    retrain_triggered = drift_share > DRIFT_SHARE_THRESHOLD
    print(f"\n  Drift share : {drift_share:.1%} (seuil : {DRIFT_SHARE_THRESHOLD:.1%})")

    if retrain_triggered:
        print("  Drift détecté — retraining déclenché automatiquement...")
        try:
            from src.models.training_v2 import train_model
            train_model()
            print("  Retraining terminé.")
        except Exception as e:
            print(f"  [WARN] Retraining échoué : {e}")
    else:
        print("  Pas de drift significatif — pas de retraining.")

    # 5. Log MLflow
    run_id = log_to_mlflow(drift_results, perf_results, retrain_triggered)

    duree = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  MONITORING TERMINÉ en {duree:.1f}s")
    print("=" * 70)

    return {
        "status":              "monitoring completed",
        "duration_seconds":    round(duree, 1),
        "split_year":          SPLIT_YEAR,
        "drift_share":         round(drift_share, 4),
        "drift_threshold":     DRIFT_SHARE_THRESHOLD,
        "retrain_triggered":   retrain_triggered,
        "mlflow_run_id":       run_id,
        "data_drift_report":   "reports/data_drift.html",
        "performance_report":  "reports/model_performance.html" if perf_results else "non généré",
    }


if __name__ == "__main__":
    run_monitoring()