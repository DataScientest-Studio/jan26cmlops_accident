# -*- coding: utf-8 -*-
"""
=============================================================================
  train_model.py - XGBoost binaire avec focus sur la detection des Tues
=============================================================================

OBJECTIF : Entrainer un XGBoost pour predire si un usager est INDEMNE ou non,
           avec un focus particulier sur la bonne detection des TUES (grav=2).

CIBLE BINAIRE :
  grav_bin = 0 : Indemne (grav=1)
  grav_bin = 1 : Reste   (grav=2 Tue + grav=3 Hospitalise + grav=4 Blesse leger)

AMELIORATIONS vs training.py de Seb :
  - Conservation des -1 (non renseigne) comme categorie valide
  - GPS 0 -> NaN (mais lignes conservees)
  - JOIN holidays (table deja en BDD mais jamais utilisee par Seb)
  - Feature engineering : age, heure, is_weekend, is_holiday
  - Sample weights manuels focus tues (x8) au lieu de "balanced" auto
  - Optimisation du seuil de decision (recall tues >= 75%)
  - Visualisations : ROC, Precision-Recall, Feature Importance, Confusion Matrix
  - Checkpoints parquet entre phases
  - Integration MLflow (meme tracking URI que Seb)

PHASES :
  1 : Chargement PostgreSQL + jointure avec holidays
  2 : Nettoyage + feature engineering + cible binaire
  3 : Cramer's V + selection de features
  4 : Entrainement XGBoost avec sample weights focus tues
  5 : Evaluation globale + focus tues
  6 : Optimisation du seuil de decision
  7 : Visualisations (ROC, PR, Feature Importance, Confusion Matrix)
  8 : MLflow tracking + model registry
  9 : Sauvegarde finale

Usage :
  python src/models/training_v2.py
  (necessite PostgreSQL avec les tables chargees via fill_database.py)

Auteur : Projet MLOps Accidents - DataScientest (Theodys)
Date   : Avril 2026
=============================================================================
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import psycopg2
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import contingency
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay,
)
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

import subprocess
import yaml

warnings.filterwarnings("ignore")

# Charger les variables d'environnement depuis .env (racine du projet)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

# ============================================================================
#  CONFIGURATION
# ============================================================================
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "src", "data")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Checkpoints
CHECKPOINT_RAW = os.path.join(DATA_DIR, "checkpoint_01_raw.parquet")
CHECKPOINT_CLEAN = os.path.join(DATA_DIR, "checkpoint_02_clean.parquet")
CHECKPOINT_MODEL = os.path.join(MODELS_DIR, "checkpoint_03_model_intermed.pkl")

# Sorties finales
MODEL_FILE = os.path.join(MODELS_DIR, "model_focus_tues.pkl")
THRESHOLD_FILE = os.path.join(MODELS_DIR, "threshold_focus_tues.pkl")
REPORT_FILE = os.path.join(REPORTS_DIR, "training_report_focus_tues.txt")
ROC_PNG = os.path.join(FIGURES_DIR, "roc_curve_focus_tues.png")
PR_PNG = os.path.join(FIGURES_DIR, "pr_curve_focus_tues.png")
FI_PNG = os.path.join(FIGURES_DIR, "feature_importance_focus_tues.png")
CM_PNG = os.path.join(FIGURES_DIR, "confusion_matrix_focus_tues.png")

# PostgreSQL (lu depuis .env, valeurs par defaut si absent)
CONN_PARAMS = {
    "dbname": os.getenv("POSTGRES_DB", "mlops_accidents"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
}

# Sample weights focus tues
WEIGHTS_GRAV = {1: 1.0, 2: 8.0, 3: 2.0, 4: 1.5}

# Hyperparametres XGBoost
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    scale_pos_weight=1,
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)

# Seuil
TARGET_RECALL_TUE = 0.75
TARGET_PREC_GLOBALE = 0.55

# MLflow (lu depuis .env)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "accidentologie-focus-tues")
MODEL_REGISTRY_NAME = os.getenv("MLFLOW_REGISTRY_NAME", "gravite-focus-tues")

# Colonnes a supprimer (identifiants, texte libre, GPS brut)
COLS_TO_DROP = [
    "num_acc", "num_veh", "id_vehicule",
    "adr", "voie", "v1", "v2",
    "lat", "long",
    "dep", "com",
    "pr", "pr1",
    "grav", "grav_bin",  # cibles, pas features
]

# Rapport texte
report_lines = []


def radd(line=""):
    """Ajoute une ligne au rapport et l'affiche."""
    print(line)
    report_lines.append(line)


# ============================================================================
#  PHASE 1 : CHARGEMENT PostgreSQL + JOINTURE HOLIDAYS
# ============================================================================
def phase1_load():
    radd("=" * 70)
    radd("  PHASE 1 : Chargement PostgreSQL + jointure holidays")
    radd("=" * 70)
    t0 = time.time()

    conn = psycopg2.connect(**CONN_PARAMS)

    users = pd.read_sql("SELECT * FROM users;", conn)
    carac = pd.read_sql("SELECT * FROM caracteristics;", conn)
    places = pd.read_sql("SELECT * FROM places;", conn)
    vehicles = pd.read_sql("SELECT * FROM vehicles;", conn)

    # Holidays - table chargee par Seb mais jamais jointe
    try:
        holidays = pd.read_sql("SELECT * FROM holidays;", conn)
        has_holidays = True
        radd(f"  holidays : {len(holidays)} lignes")
    except Exception:
        has_holidays = False
        radd("  [WARN] Table holidays absente, on continue sans.")

    conn.close()

    radd(f"  users        : {len(users):>10,} lignes")
    radd(f"  carac        : {len(carac):>10,} lignes")
    radd(f"  places       : {len(places):>10,} lignes")
    radd(f"  vehicles     : {len(vehicles):>10,} lignes")

    # Normaliser les noms de colonnes en minuscules
    users.columns = users.columns.str.lower()
    carac.columns = carac.columns.str.lower()
    places.columns = places.columns.str.lower()
    vehicles.columns = vehicles.columns.str.lower()

    # Jointure principale (comme Seb)
    df = (
        users.merge(vehicles, on=["num_acc", "num_veh"], how="left")
        .merge(carac, on="num_acc", how="left")
        .merge(places, on="num_acc", how="left")
    )
    radd(f"  Apres jointure principale : {len(df):,} lignes x {df.shape[1]} colonnes")

    # LEFT JOIN holidays si disponible
    if has_holidays:
        holidays.columns = holidays.columns.str.lower()
        # Construire une colonne date dans df pour matcher holidays
        if "an" in df.columns and "mois" in df.columns and "jour" in df.columns:
            df["date_acc"] = pd.to_datetime(
                df["an"].astype(str) + "-" + df["mois"].astype(str).str.zfill(2) + "-" + df["jour"].astype(str).str.zfill(2),
                errors="coerce",
            )
            if "ds" in holidays.columns:
                holidays["ds"] = pd.to_datetime(holidays["ds"], errors="coerce")
                holidays["is_holiday"] = 1
                df = df.merge(holidays[["ds", "is_holiday"]], left_on="date_acc", right_on="ds", how="left")
                df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
                df.drop(columns=["ds", "date_acc"], inplace=True, errors="ignore")
                radd(f"  Jours feries trouves : {df['is_holiday'].sum():,}")
            else:
                df.drop(columns=["date_acc"], inplace=True, errors="ignore")
                df["is_holiday"] = 0
        else:
            df["is_holiday"] = 0
    else:
        df["is_holiday"] = 0

    df = df.drop_duplicates()
    radd(f"  Apres dedup : {len(df):,} lignes")

    # Checkpoint
    df.to_parquet(CHECKPOINT_RAW, index=False)
    radd(f"  Checkpoint sauve : {CHECKPOINT_RAW}")
    radd(f"  Duree phase 1 : {time.time() - t0:.1f}s")
    return df


# ============================================================================
#  PHASE 2 : NETTOYAGE + FEATURE ENGINEERING + CIBLE BINAIRE
# ============================================================================
def phase2_clean(df):
    radd("")
    radd("=" * 70)
    radd("  PHASE 2 : Nettoyage + feature engineering + cible binaire")
    radd("=" * 70)
    t0 = time.time()

    n_avant = len(df)

    # --- GPS : 0 -> NaN (on garde les lignes) ---
    for col_gps in ["lat", "long"]:
        if col_gps in df.columns:
            df[col_gps] = df[col_gps].replace(0, np.nan)
            n_zero = df[col_gps].isna().sum()
            radd(f"  {col_gps} : {n_zero:,} valeurs 0 -> NaN (lignes conservees)")

    # --- Valeurs "-" -> NaN -> -1 (au lieu de dropna comme Seb) ---
    for col in df.select_dtypes(include="object").columns:
        mask_tiret = df[col] == "-"
        if mask_tiret.sum() > 0:
            radd(f"  {col} : {mask_tiret.sum():,} valeurs '-' converties en -1")
            df.loc[mask_tiret, col] = "-1"

    # --- Feature engineering ---
    # Age
    if "an_nais" in df.columns and "an" in df.columns:
        df["an_nais"] = pd.to_numeric(df["an_nais"], errors="coerce")
        df["an"] = pd.to_numeric(df["an"], errors="coerce")
        df["age"] = df["an"] - df["an_nais"]
        df.loc[df["age"] < 0, "age"] = np.nan
        df.loc[df["age"] > 120, "age"] = np.nan
        radd(f"  Feature 'age' creee (mediane = {df['age'].median():.0f})")

    # Heure
    if "hrmn" in df.columns:
        df["hrmn"] = df["hrmn"].astype(str).str.replace(":", "").str.zfill(4)
        df["heure"] = pd.to_numeric(df["hrmn"].str[:2], errors="coerce")
        radd(f"  Feature 'heure' creee (0-23)")

    # is_weekend
    if "an" in df.columns and "mois" in df.columns and "jour" in df.columns:
        date_tmp = pd.to_datetime(
            df["an"].astype(str) + "-" + df["mois"].astype(str).str.zfill(2) + "-" + df["jour"].astype(str).str.zfill(2),
            errors="coerce",
        )
        df["is_weekend"] = date_tmp.dt.dayofweek.isin([5, 6]).astype(int)
        radd(f"  Feature 'is_weekend' creee ({df['is_weekend'].sum():,} weekend)")

    # --- Cible binaire ---
    df["grav"] = pd.to_numeric(df["grav"], errors="coerce")
    df = df.dropna(subset=["grav"])
    df["grav"] = df["grav"].astype(int)
    df["grav_bin"] = (df["grav"] != 1).astype(int)  # 0=indemne, 1=reste

    radd(f"  Cible binaire :")
    radd(f"    grav_bin=0 (Indemne) : {(df['grav_bin'] == 0).sum():>10,}")
    radd(f"    grav_bin=1 (Reste)   : {(df['grav_bin'] == 1).sum():>10,}")
    radd(f"      dont Tues (grav=2) : {(df['grav'] == 2).sum():>10,}")
    radd(f"      dont Hospit (grav=3): {(df['grav'] == 3).sum():>10,}")
    radd(f"      dont Legers (grav=4): {(df['grav'] == 4).sum():>10,}")

    radd(f"  Lignes finales : {len(df):,} (vs {n_avant:,} avant nettoyage)")

    # --- Encodage des colonnes object ---
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(-1)
        df[col] = df[col].astype(int)

    # Remplir les NaN restants
    df = df.fillna(-1)

    # Checkpoint
    df.to_parquet(CHECKPOINT_CLEAN, index=False)
    radd(f"  Checkpoint sauve : {CHECKPOINT_CLEAN}")
    radd(f"  Duree phase 2 : {time.time() - t0:.1f}s")
    return df


# ============================================================================
#  PHASE 3 : CRAMER'S V - CORRELATIONS ENTRE FEATURES ET CIBLE
# ============================================================================
def cramers_v(x, y):
    """Calcul du V de Cramer entre deux Series."""
    ct = pd.crosstab(x, y)
    res = contingency.association(ct.values, method="cramer")
    return res


def phase3_cramers(df):
    radd("")
    radd("=" * 70)
    radd("  PHASE 3 : Correlations Cramer's V avec grav_bin")
    radd("=" * 70)
    t0 = time.time()

    # Identifier les features candidates
    cols_feat = [c for c in df.columns if c not in COLS_TO_DROP and c != "grav_bin"]
    results = {}
    for col in cols_feat:
        try:
            v = cramers_v(df[col], df["grav_bin"])
            results[col] = round(v, 4)
        except Exception:
            pass

    results_sorted = sorted(results.items(), key=lambda x: x[1], reverse=True)

    radd(f"  {'Variable':<25} {'Cramer V':>10}")
    radd(f"  {'-' * 25} {'-' * 10}")
    for var, v in results_sorted[:20]:
        radd(f"  {var:<25} {v:>10.4f}")

    # Garder les features avec V > 0.01
    selected = [var for var, v in results_sorted if v > 0.01]
    radd(f"\n  Features selectionnees (V > 0.01) : {len(selected)}")
    radd(f"  Duree phase 3 : {time.time() - t0:.1f}s")
    return selected


# ============================================================================
#  PHASE 4 : ENTRAINEMENT XGBOOST AVEC SAMPLE WEIGHTS
# ============================================================================
def phase4_train(df, selected_features):
    radd("")
    radd("=" * 70)
    radd("  PHASE 4 : Entrainement XGBoost avec sample weights focus tues")
    radd("=" * 70)
    t0 = time.time()

    # Features et cible
    feature_cols = [c for c in selected_features if c in df.columns and c not in COLS_TO_DROP]
    X = df[feature_cols].copy()
    y = df["grav_bin"].copy()
    grav_orig = df["grav"].copy()  # pour les weights

    radd(f"  Features : {len(feature_cols)} colonnes")
    radd(f"  Echantillons : {len(X):,}")

    # Split
    X_train, X_test, y_train, y_test, grav_train, grav_test = train_test_split(
        X, y, grav_orig, test_size=0.2, random_state=42, stratify=y
    )

    radd(f"  Train : {len(X_train):,} | Test : {len(X_test):,}")

    # Sample weights
    w_train = grav_train.map(WEIGHTS_GRAV).fillna(1.0).values
    radd(f"  Poids : Tue=x{WEIGHTS_GRAV[2]}, Hospit=x{WEIGHTS_GRAV[3]}, Leger=x{WEIGHTS_GRAV[4]}, Indemne=x{WEIGHTS_GRAV[1]}")

    # Modele
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, sample_weight=w_train)

    # Checkpoint intermediaire
    joblib.dump(model, CHECKPOINT_MODEL)
    radd(f"  Checkpoint modele : {CHECKPOINT_MODEL}")
    radd(f"  Duree phase 4 : {time.time() - t0:.1f}s")

    return model, X_train, X_test, y_train, y_test, grav_train, grav_test, feature_cols


# ============================================================================
#  PHASE 5 : EVALUATION GLOBALE + FOCUS TUES
# ============================================================================
def phase5_eval(model, X_test, y_test, grav_test):
    radd("")
    radd("=" * 70)
    radd("  PHASE 5 : Evaluation globale + focus Tues")
    radd("=" * 70)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    radd(f"\n  --- Metriques globales (seuil=0.5) ---")
    radd(f"  Accuracy  : {acc:.4f}")
    radd(f"  F1 macro  : {f1:.4f}")
    radd(f"  Precision : {prec:.4f}")
    radd(f"  Recall    : {rec:.4f}")
    radd(f"  AUC-ROC   : {auc:.4f}")

    radd(f"\n  --- Classification Report ---")
    radd(classification_report(y_test, y_pred, target_names=["Indemne", "Reste"]))

    # Focus Tues
    mask_tue = grav_test == 2
    if mask_tue.sum() > 0:
        tue_pred = y_pred[mask_tue]
        tue_true = y_test.values[mask_tue]
        recall_tue = recall_score(tue_true, tue_pred)
        radd(f"  --- Focus TUES (grav=2) ---")
        radd(f"  Tues dans le test set : {mask_tue.sum()}")
        radd(f"  Tues detectes (pred=1) : {tue_pred.sum()}")
        radd(f"  Recall Tues (seuil=0.5) : {recall_tue:.4f}")
    else:
        recall_tue = 0.0

    return y_proba, acc, f1, auc, recall_tue


# ============================================================================
#  PHASE 6 : OPTIMISATION DU SEUIL DE DECISION
# ============================================================================
def phase6_threshold(model, X_test, y_test, grav_test, y_proba):
    radd("")
    radd("=" * 70)
    radd("  PHASE 6 : Optimisation du seuil (recall Tues >= 75%)")
    radd("=" * 70)

    mask_tue = grav_test == 2
    best_threshold = 0.5
    best_f1 = 0.0

    for t in np.arange(0.10, 0.90, 0.01):
        y_t = (y_proba >= t).astype(int)

        # Recall sur les tues
        if mask_tue.sum() > 0:
            r_tue = recall_score(y_test.values[mask_tue], y_t[mask_tue])
        else:
            r_tue = 0.0

        # Precision globale
        p_glob = precision_score(y_test, y_t, zero_division=0)

        if r_tue >= TARGET_RECALL_TUE and p_glob >= TARGET_PREC_GLOBALE:
            f1_t = f1_score(y_test, y_t, average="macro")
            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = round(t, 2)

    # Appliquer le seuil optimal
    y_opt = (y_proba >= best_threshold).astype(int)
    acc_opt = accuracy_score(y_test, y_opt)
    f1_opt = f1_score(y_test, y_opt, average="macro")
    prec_opt = precision_score(y_test, y_opt)
    rec_opt = recall_score(y_test, y_opt)

    if mask_tue.sum() > 0:
        rec_tue_opt = recall_score(y_test.values[mask_tue], y_opt[mask_tue])
    else:
        rec_tue_opt = 0.0

    radd(f"  Seuil optimal : {best_threshold}")
    radd(f"  Accuracy      : {acc_opt:.4f}")
    radd(f"  F1 macro      : {f1_opt:.4f}")
    radd(f"  Precision     : {prec_opt:.4f}")
    radd(f"  Recall global : {rec_opt:.4f}")
    radd(f"  Recall Tues   : {rec_tue_opt:.4f}")

    radd(f"\n  --- Classification Report (seuil={best_threshold}) ---")
    radd(classification_report(y_test, y_opt, target_names=["Indemne", "Reste"]))

    return best_threshold, f1_opt, rec_tue_opt


# ============================================================================
#  PHASE 7 : VISUALISATIONS
# ============================================================================
def phase7_plots(model, X_test, y_test, y_proba, best_threshold, feature_cols):
    radd("")
    radd("=" * 70)
    radd("  PHASE 7 : Visualisations")
    radd("=" * 70)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC - Focus Tues\nIndemne vs Reste", fontsize=12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ROC_PNG, dpi=150)
    plt.close(fig)
    radd(f"  ROC curve : {ROC_PNG}")

    # Precision-Recall Curve
    prec_arr, rec_arr, thresholds_pr = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec_arr, prec_arr, "g-", linewidth=2, label=f"AP = {ap:.4f}")
    # Marquer le seuil optimal
    idx_opt = np.argmin(np.abs(thresholds_pr - best_threshold))
    ax.plot(rec_arr[idx_opt], prec_arr[idx_opt], "r*", markersize=15, label=f"Seuil = {best_threshold}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Courbe Precision-Recall - Focus Tues", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PR_PNG, dpi=150)
    plt.close(fig)
    radd(f"  PR curve  : {PR_PNG}")

    # Feature Importance
    importances = model.feature_importances_
    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    top20 = fi.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    top20.sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_title("Top 20 variables - XGBoost Focus Tues\n(sample weights focus Tues)", fontsize=12)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(FI_PNG, dpi=150)
    plt.close(fig)
    radd(f"  Feature importance : {FI_PNG}")

    # Confusion Matrix (seuil optimal)
    y_opt = (y_proba >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_opt)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Indemne", "Reste"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Matrice de confusion (seuil={best_threshold})", fontsize=12)
    fig.tight_layout()
    fig.savefig(CM_PNG, dpi=150)
    plt.close(fig)
    radd(f"  Confusion matrix : {CM_PNG}")


# ============================================================================
#  PHASE 8 : MLFLOW TRACKING + MODEL REGISTRY
# ============================================================================
def phase8_mlflow(model, acc, f1, auc, recall_tue, best_threshold, f1_opt, rec_tue_opt, X_train, X_test, dvc_hashes):
    radd("")
    radd("=" * 70)
    radd("  PHASE 8 : MLflow tracking")
    radd("=" * 70)

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        client = MlflowClient()

        with mlflow.start_run() as run:
            run_id = run.info.run_id

            # Params
            for k, v in XGB_PARAMS.items():
                mlflow.log_param(k, v)
            mlflow.log_param("n_train",           len(X_train))
            mlflow.log_param("n_test",            len(X_test))
            mlflow.log_param("target",            "binaire_focus_tues")
            mlflow.log_param("weight_tue",        WEIGHTS_GRAV[2])
            mlflow.log_param("weight_hospit",     WEIGHTS_GRAV[3])
            mlflow.log_param("weight_leger",      WEIGHTS_GRAV[4])
            mlflow.log_param("threshold_optimal", best_threshold)

            # Enregistrement Hash DVC
            for label, hash_val in dvc_hashes.items():
                mlflow.log_param(f"dvc_hash_{label}", hash_val)

            # Métriques (seuil 0.5)
            mlflow.log_metric("accuracy",        round(float(acc), 4))
            mlflow.log_metric("f1_macro",        round(float(f1), 4))
            mlflow.log_metric("auc_roc",         round(float(auc), 4))
            mlflow.log_metric("recall_tues_0.5", round(float(recall_tue), 4))

            # Métriques (seuil optimal)
            mlflow.log_metric("f1_macro_opt",    round(float(f1_opt), 4))
            mlflow.log_metric("recall_tues_opt", round(float(rec_tue_opt), 4))
            mlflow.log_metric("threshold",       round(float(best_threshold), 4))

            # Modèle
            mlflow.xgboost.log_model(model, artifact_path="model")

            # Artefacts
            for png in [ROC_PNG, PR_PNG, FI_PNG, CM_PNG]:
                if os.path.exists(png):
                    mlflow.log_artifact(png)
            if os.path.exists(REPORT_FILE):
                mlflow.log_artifact(REPORT_FILE)

            # Model Registry + comparaison
            mv          = mlflow.register_model(f"runs:/{run_id}/model", MODEL_REGISTRY_NAME)
            new_version = mv.version

            prod_versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["Production"])
            if not prod_versions:
                client.transition_model_version_stage(MODEL_REGISTRY_NAME, new_version, "Production")
                radd(f"  [Registry] v{new_version} -> Production (premier modele)")
            else:
                prod_f1 = client.get_run(prod_versions[0].run_id).data.metrics.get("f1_macro_opt", 0.0)
                delta   = f1_opt - prod_f1
                mlflow.log_metric("delta_f1", round(delta, 4))
                if f1_opt > prod_f1:
                    client.transition_model_version_stage(MODEL_REGISTRY_NAME, prod_versions[0].version, "Archived")
                    client.transition_model_version_stage(MODEL_REGISTRY_NAME, new_version, "Production")
                    radd(f"  [Registry] v{new_version} -> Production (+{delta:.4f})")
                else:
                    client.transition_model_version_stage(MODEL_REGISTRY_NAME, new_version, "Staging")
                    radd(f"  [Registry] v{new_version} -> Staging ({delta:.4f})")

        radd(f"  [MLflow] Run ID : {run_id}")
        radd(f"  [MLflow] UI -> {MLFLOW_TRACKING_URI}")

    except Exception as e:
        radd(f"  [WARN] MLflow indisponible ({e})")
        radd(f"  Le modele est sauvegarde localement, MLflow sera mis a jour au prochain run.")


# ============================================================================
#  PHASE 9 : SAUVEGARDE FINALE
# ============================================================================
def phase9_save(model, best_threshold, feature_cols):
    radd("")
    radd("=" * 70)
    radd("  PHASE 9 : Sauvegarde finale")
    radd("=" * 70)

    joblib.dump(model, MODEL_FILE)
    radd(f"  Modele   : {MODEL_FILE}")

    joblib.dump({"threshold": best_threshold, "features": feature_cols}, THRESHOLD_FILE)
    radd(f"  Seuil    : {THRESHOLD_FILE}")

    # Rapport texte
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    radd(f"  Rapport  : {REPORT_FILE}")

    radd("")
    radd("=" * 70)
    radd("  TERMINE - Prediction via : python src/models/predict_model.py")
    radd("=" * 70)

    # DVC : tracking modèles et autres fichiers
    dvc_hashes = {}
    dvc_files = {
        "model":     MODEL_FILE,
        "threshold": THRESHOLD_FILE,
        "report":    REPORT_FILE,
    }
    for label, filepath in dvc_files.items():
        try:
            subprocess.run(["dvc", "add", filepath], check=True, capture_output=True)
            dvc_file = filepath + ".dvc"
            with open(dvc_file, "r") as f:
                dvc_meta = yaml.safe_load(f)
            dvc_hashes[label] = dvc_meta["outs"][0]["md5"]
            radd(f"  DVC [{label}] hash : {dvc_hashes[label]}")
        except Exception as e:
            radd(f"  [WARN] DVC indisponible pour {label} : {e}")
            dvc_hashes[label] = "unavailable"

    # DVC : tracking datasets
    dataset_files = {
        "caracteristics": os.path.join(DATA_DIR, "caracteristics.csv"),
        "places":         os.path.join(DATA_DIR, "places.csv"),
        "vehicles":       os.path.join(DATA_DIR, "vehicles.csv"),
        "users":          os.path.join(DATA_DIR, "users.csv"),
    }
    for label, filepath in dataset_files.items():
        if not os.path.exists(filepath):
            radd(f"  [WARN] Dataset absent pour DVC : {filepath}")
            dvc_hashes[f"data_{label}"] = "unavailable"
            continue
        try:
            subprocess.run(["dvc", "add", filepath], check=True, capture_output=True)
            dvc_file = filepath + ".dvc"
            with open(dvc_file, "r") as f:
                dvc_meta = yaml.safe_load(f)
            dvc_hashes[f"data_{label}"] = dvc_meta["outs"][0]["md5"]
            radd(f"  DVC [data_{label}] hash : {dvc_hashes[f'data_{label}']}")
        except Exception as e:
            radd(f"  [WARN] DVC indisponible pour {label} : {e}")
            dvc_hashes[f"data_{label}"] = "unavailable"

    # DVC push vers le remote
    try:
        subprocess.run(["dvc", "push"], check=True, capture_output=True)
        radd("  DVC push : OK")
    except Exception as e:
        radd(f"  [WARN] DVC push echoue : {e}")

    radd("")
    radd("=" * 70)
    radd("  TERMINE - Prediction via : python src/models/predict_model.py")
    radd("=" * 70)

    return dvc_hashes

# Pour exporter les données de postgresql en csv (pour stockage DVC)
def export_datasets_to_csv(df_raw: pd.DataFrame) -> None:
    """Exporte les tables brutes en CSV pour le tracking DVC."""
    os.makedirs(DATA_DIR, exist_ok=True)

    conn = psycopg2.connect(**CONN_PARAMS)
    tables = ["caracteristics", "places", "vehicles", "users"]
    for table in tables:
        csv_path = os.path.join(DATA_DIR, f"{table}.csv")
        if not os.path.exists(csv_path):  # export une seule fois si absent
            pd.read_sql(f"SELECT * FROM {table};", conn).to_csv(csv_path, index=False)
            radd(f"  Export CSV : {csv_path}")
        else:
            radd(f"  CSV deja present : {csv_path}")
    conn.close()

# ============================================================================
#  MAIN
# ============================================================================
def train_model():
    """Point d'entree principal - compatible avec api.py."""
    radd("")
    radd("=" * 70)
    radd("  TRAINING_V2.PY - XGBoost binaire focus Tues")
    radd("=" * 70)

    # Phase 1
    df = phase1_load()
    export_datasets_to_csv(df)

    # Phase 2
    df = phase2_clean(df)

    # Phase 3
    selected = phase3_cramers(df)

    # Phase 4
    model, X_train, X_test, y_train, y_test, grav_train, grav_test, feature_cols = phase4_train(df, selected)

    # Phase 5
    y_proba, acc, f1, auc, recall_tue = phase5_eval(model, X_test, y_test, grav_test)

    # Phase 6
    best_threshold, f1_opt, rec_tue_opt = phase6_threshold(model, X_test, y_test, grav_test, y_proba)

    # Phase 7
    phase7_plots(model, X_test, y_test, y_proba, best_threshold, feature_cols)

    #Phase 9 (sauvegarde avant MLflow pour pouvoir log les artefacts)
    dvc_hashes = phase9_save(model, best_threshold, feature_cols)

    # Phase 8
    phase8_mlflow(model, acc, f1, auc, recall_tue, best_threshold, f1_opt, rec_tue_opt, X_train, X_test, dvc_hashes) 

    return {
        "status": "training completed",
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "auc_roc": float(auc),
        "recall_tues": float(recall_tue),
        "threshold": float(best_threshold),
        "f1_opt": float(f1_opt),
        "recall_tues_opt": float(rec_tue_opt),
    }


if __name__ == "__main__":
    result = train_model()
    print("\n", result)
