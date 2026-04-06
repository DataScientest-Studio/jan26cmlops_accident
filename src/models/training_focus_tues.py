# -*- coding: utf-8 -*-
"""
=============================================================================
  training_focus_tues.py — XGBoost avec focus sur la détection des Tués
=============================================================================

OBJECTIF : Entraîner un XGBoost pour prédire si un usager est INDEMNE ou non,
           avec un focus particulier sur la bonne détection des TUÉS (grav=2).

CIBLE :
  grav_bin = 0 : Indemne (grav=1)
  grav_bin = 1 : Reste   (grav=2 Tué + grav=3 Hospitalisé + grav=4 Blessé léger)

STRATÉGIE POUR DÉTECTER LES TUÉS :
  On utilise des SAMPLE WEIGHTS pour sur-pondérer les tués pendant l'entraînement.
  → Chaque erreur sur un "tué" coûte 8x plus cher au modèle qu'une erreur sur un indemne.
  → Le modèle apprend à être plus prudent sur les cas ressemblant à des tués.

  Poids par catégorie de gravité :
    grav=1 (Indemne)      : poids = 1.0   (référence)
    grav=2 (Tué)          : poids = 8.0   (sur-pondéré ×8)
    grav=3 (Hospitalisé)  : poids = 2.0   (intermédiaire)
    grav=4 (Blessé léger) : poids = 1.5   (légèrement sur-pondéré)

OPTIMISATION DU SEUIL :
  On cherche le seuil qui maximise le recall spécifiquement sur les TUÉS,
  pas seulement sur la classe 1 globale.

PHASES :
  PHASE 1 : Chargement + préparation + sample weights
  PHASE 2 : Corrélations Cramér's V
  PHASE 3 : Entraînement XGBoost + checkpoint modèle intermédiaire
  PHASE 4 : Évaluation globale et focus Tués
  PHASE 5 : Optimisation du seuil de décision (recall Tués)
  PHASE 6 : Visualisations (ROC, Precision-Recall, Feature Importance)
  PHASE 7 : Sauvegarde finale

Usage :
  python training_focus_tues.py
  (nécessite d'avoir lancé preprocess_focus_tues.py auparavant)

Auteur : Projet MLOps Accidents — DataScientest
Date   : Avril 2026
=============================================================================
"""

import os
import sys
import io
import time
import warnings
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ['PYTHONIOENCODING'] = 'utf-8'
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from scipy.stats import chi2_contingency
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

# =============================================================================
# CONFIGURATION
# =============================================================================

# Racine du repo = 2 niveaux au-dessus de src/models/
REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR       = os.path.join(REPO_ROOT, "src", "data")
MODELS_DIR     = os.path.join(REPO_ROOT, "models")
REPORTS_DIR    = os.path.join(REPO_ROOT, "reports")
FIGURES_DIR    = os.path.join(REPORTS_DIR, "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Fichiers d'entrée
PARQUET_FINAL  = os.path.join(DATA_DIR, "dataset_focus_tues.parquet")

# Checkpoints et fichiers de sortie
CHECKPOINT_MODEL_INTERMED = os.path.join(MODELS_DIR, "checkpoint_03_model_intermed.pkl")
MODEL_FILE     = os.path.join(MODELS_DIR,  "model_focus_tues.pkl")
THRESHOLD_FILE = os.path.join(MODELS_DIR,  "threshold_focus_tues.pkl")
REPORT_FILE    = os.path.join(REPORTS_DIR, "training_report_focus_tues.txt")
ROC_PNG        = os.path.join(FIGURES_DIR, "roc_curve_focus_tues.png")
PR_PNG         = os.path.join(FIGURES_DIR, "pr_curve_focus_tues.png")
FI_PNG         = os.path.join(FIGURES_DIR, "feature_importance_focus_tues.png")
CM_PNG         = os.path.join(FIGURES_DIR, "confusion_matrix_focus_tues.png")

TEST_SIZE      = 0.20
RANDOM_STATE   = 42
TARGET_RECALL_TUE  = 0.75   # Objectif : détecter 75% des tués
TARGET_PREC_GLOBALE = 0.55  # Objectif : precision globale classe 1 ≥ 55%

# Poids par catégorie (grav original)
# POURQUOI ces valeurs ?
#   - Un tué non détecté = erreur très grave (irréversible)
#   - Un hospitalisé non détecté = erreur grave
#   - Un blessé léger non détecté = erreur modérée
#   - Un indemne mal classé = fausse alarme (acceptable)
WEIGHTS = {1: 1.0, 2: 8.0, 3: 2.0, 4: 1.5}

# Colonnes continues (peuvent contenir des NaN → imputation par médiane)
COLS_FLOAT = ["lat", "long", "lartpc", "larrout"]

print("=" * 70)
print("  TRAINING focus_tues — XGBoost focus Tués")
print("=" * 70)

report_lines = []

def radd(line=""):
    report_lines.append(line)
    print(line)

# =============================================================================
# PHASE 1 — CHARGEMENT + PRÉPARATION + SAMPLE WEIGHTS
# =============================================================================
print("\n[PHASE 1] Chargement et préparation du dataset...")

t0 = time.time()

if not os.path.exists(PARQUET_FINAL):
    print(f"  ✗ Fichier introuvable : {PARQUET_FINAL}")
    print("    Lancez d'abord : python preprocess_focus_tues.py")
    sys.exit(1)

df = pd.read_parquet(PARQUET_FINAL)
elapsed = time.time() - t0
print(f"  ✓ Dataset chargé : {len(df):,} lignes × {df.shape[1]} colonnes ({elapsed:.1f}s)")

# Vérification de la présence des colonnes requises
for col_required in ["grav_bin", "grav"]:
    if col_required not in df.columns:
        print(f"  ✗ Colonne manquante : '{col_required}'")
        print("    Relancez preprocess_focus_tues.py")
        sys.exit(1)

# Statistiques sur la cible
n_total     = len(df)
n_indemne   = (df["grav_bin"] == 0).sum()
n_reste     = (df["grav_bin"] == 1).sum()
n_tue       = (df["grav"] == 2).sum()
n_hospit    = (df["grav"] == 3).sum()
n_blesse    = (df["grav"] == 4).sum()

print(f"\n  Distribution grav_bin :")
print(f"    grav_bin=0 Indemne    : {n_indemne:,}  ({n_indemne/n_total*100:.1f}%)")
print(f"    grav_bin=1 Reste      : {n_reste:,}  ({n_reste/n_total*100:.1f}%)")
print(f"      dont Tués (grav=2)  : {n_tue:,}  ({n_tue/n_total*100:.1f}%)")
print(f"      dont Hospit (grav=3): {n_hospit:,}  ({n_hospit/n_total*100:.1f}%)")
print(f"      dont Blessés (grav=4): {n_blesse:,}  ({n_blesse/n_total*100:.1f}%)")
print(f"    Ratio reste/indemne   : {n_reste/n_indemne:.2f}x")

# Séparation X / y / grav_original
# grav_original conservé pour construire les sample_weights et évaluer les tués
grav_original = df["grav"].copy()
X_full = df.drop(columns=["grav_bin", "grav"])
y_full = df["grav_bin"]

print(f"\n  Variables d'entrée (X) : {X_full.shape[1]} colonnes")

# Imputation des NaN sur les colonnes continues
for col in COLS_FLOAT:
    if col in X_full.columns:
        n_nan = X_full[col].isna().sum()
        if n_nan > 0:
            mediane = X_full[col].median()
            X_full[col] = X_full[col].fillna(mediane)
            print(f"    {col} : {n_nan:,} NaN imputés par la médiane ({mediane:.4f})")

# Division train/test (stratifiée sur grav_bin)
X_train, X_test, y_train, y_test, grav_train, grav_test = train_test_split(
    X_full, y_full, grav_original,
    test_size=TEST_SIZE,
    stratify=y_full,
    random_state=RANDOM_STATE
)

print(f"\n  Division train/test :")
print(f"    Train : {len(X_train):,} lignes")
print(f"    Test  : {len(X_test):,} lignes")

# Calcul des sample weights pour le train
# Chaque usager reçoit un poids selon sa gravité RÉELLE (grav original)
# Cela sur-pondère les tués dans la fonction de perte du XGBoost
sample_weights_train = grav_train.map(WEIGHTS).values
print(f"\n  Sample weights (par gravité) :")
for g, w in WEIGHTS.items():
    n = (grav_train == g).sum()
    print(f"    grav={g} ({['','Indemne','Tué','Hospitalisé','Blessé léger'][g]:<15}) : {w:.1f} × {n:,} = poids total {w*n:,.0f}")

print(f"\n  Poids total train : {sample_weights_train.sum():,.0f}")
print(f"  Poids moyen       : {sample_weights_train.mean():.3f}")

# =============================================================================
# PHASE 2 — CORRÉLATIONS CRAMÉR'S V
# =============================================================================
print("\n[PHASE 2] Corrélations Cramér's V avec grav_bin...")

COLS_CONTINUES = {"lat", "long", "age", "lartpc", "larrout", "occutc", "nbv"}
cols_cat = [c for c in X_full.columns if c not in COLS_CONTINUES]

def cramers_v(col1, col2):
    tableau = pd.crosstab(col1, col2)
    chi2, _, _, _ = chi2_contingency(tableau)
    n = tableau.sum().sum()
    k = min(tableau.shape)
    if k <= 1 or n == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * (k - 1))))

N_SAMPLE = min(100_000, len(df))
print(f"  Calcul sur {N_SAMPLE:,} lignes (échantillon pour accélération)...")
df_sample = df.sample(n=N_SAMPLE, random_state=RANDOM_STATE, replace=False)

resultats_cv = []
for col in cols_cat:
    if col in df_sample.columns:
        try:
            masque = df_sample[col].notna() & df_sample["grav_bin"].notna()
            v = cramers_v(df_sample.loc[masque, col], df_sample.loc[masque, "grav_bin"])
            resultats_cv.append((col, v))
        except Exception:
            pass

resultats_cv.sort(key=lambda x: x[1], reverse=True)

radd("\n" + "=" * 68)
radd(f"  RAPPORT D'ENTRAÎNEMENT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
radd(f"  Modèle : XGBoost | Cible : indemne (0) vs reste (1) | Focus : Tués")
radd("=" * 68)

radd("\n[CORRÉLATIONS CRAMÉR'S V — variables vs grav_bin]")
radd(f"  {'Rang':<5}  {'Variable':<12}  {'Cramér V':>9}  Interprétation")
radd(f"  {'─'*5}  {'─'*12}  {'─'*9}  {'─'*20}")

for rang, (col, v) in enumerate(resultats_cv, start=1):
    interp = "Forte" if v >= 0.2 else ("Modérée" if v >= 0.1 else ("Faible" if v >= 0.05 else "Très faible"))
    radd(f"  {rang:<5}  {col:<12}  {v:>9.4f}  {interp}")
    if rang >= 15:
        break

top5 = [col for col, _ in resultats_cv[:5]]
radd(f"\n  Top 5 : {', '.join(top5)}")

# =============================================================================
# PHASE 3 — ENTRAÎNEMENT XGBOOST + CHECKPOINT
# =============================================================================
# XGBoost avec sample_weight pour sur-pondérer les tués.
#
# HYPERPARAMÈTRES :
#   n_estimators=400   : 400 arbres séquentiels (chaque arbre corrige le précédent)
#   max_depth=6        : profondeur modérée (évite le surapprentissage)
#   learning_rate=0.05 : apprentissage lent = plus précis
#   subsample=0.8      : 80% des données par arbre (régularisation)
#   colsample_bytree=0.8 : 80% des variables par arbre (régularisation)
#   scale_pos_weight   : ratio de déséquilibre (complément aux sample_weights)
#   min_child_weight=5 : au moins 5 exemples par feuille (évite surapprentissage)
print("\n[PHASE 3] Entraînement XGBoost...")

scale_pw = n_indemne / n_reste  # < 1 car reste > indemne dans cette version
# NOTA : avec sample_weights, scale_pos_weight est secondaire
# On garde scale_pos_weight à 1 pour ne pas doubler la correction
print(f"  Ratio indemne/reste : {n_indemne/n_reste:.3f}  (scale_pos_weight ignoré car sample_weights actifs)")

xgb_model = XGBClassifier(
    n_estimators     = 400,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    scale_pos_weight = 1,       # Géré par sample_weight
    eval_metric      = "logloss",
    tree_method      = "hist",
    n_jobs           = -1,
    random_state     = RANDOM_STATE,
    verbosity        = 0,
)

t_train = time.time()
xgb_model.fit(
    X_train, y_train,
    sample_weight=sample_weights_train   # Poids par usager (focus Tués)
)
temps_train = time.time() - t_train
print(f"  ✓ Entraînement terminé en {temps_train:.1f}s")

# CHECKPOINT 3 — modèle intermédiaire (avant optimisation du seuil)
joblib.dump(xgb_model, CHECKPOINT_MODEL_INTERMED, compress=3)
print(f"  ✓ Checkpoint modèle intermédiaire : {CHECKPOINT_MODEL_INTERMED}")

# Prédictions avec seuil par défaut (0.50)
y_pred_50   = xgb_model.predict(X_test)
y_proba     = xgb_model.predict_proba(X_test)[:, 1]

# =============================================================================
# PHASE 4 — ÉVALUATION GLOBALE + FOCUS TUÉS
# =============================================================================
print("\n[PHASE 4] Évaluation du modèle...")

# Métriques globales (classe 1 = tout le "reste")
acc        = accuracy_score(y_test, y_pred_50)
recall_g1  = recall_score(y_test, y_pred_50, pos_label=1, zero_division=0)
prec_g1    = precision_score(y_test, y_pred_50, pos_label=1, zero_division=0)
f1_g1      = f1_score(y_test, y_pred_50, pos_label=1, zero_division=0)
roc_auc    = roc_auc_score(y_test, y_proba)
pr_auc     = average_precision_score(y_test, y_proba)

radd("\n[ÉVALUATION — SEUIL PAR DÉFAUT 0.50]")
radd(f"  Accuracy            : {acc*100:.1f}%")
radd(f"  Recall classe 1     : {recall_g1*100:.1f}%   (% de 'reste' détectés)")
radd(f"  Precision classe 1  : {prec_g1*100:.1f}%")
radd(f"  F1 classe 1         : {f1_g1*100:.1f}%")
radd(f"  ROC-AUC             : {roc_auc:.4f}")
radd(f"  PR-AUC              : {pr_auc:.4f}")
radd(f"  Durée entraînement  : {temps_train:.1f}s")

# Rapport de classification détaillé
radd("\n[RAPPORT DE CLASSIFICATION — SEUIL 0.50]")
rapport = classification_report(
    y_test, y_pred_50,
    target_names=["Indemne (0)", "Reste (1)"]
)
radd(rapport)

# Focus sur les tués (grav=2) dans le jeu de test
# Les tués font partie de la classe 1 (grav_bin=1)
# On mesure le recall sur les tués spécifiquement :
#   parmi les vrais tués dans le test, combien le modèle prédit-il comme "classe 1" ?
masque_tue_test = (grav_test == 2)
n_tue_test      = masque_tue_test.sum()

if n_tue_test > 0:
    # Un tué est bien détecté si le modèle prédit grav_bin=1 (ce qui est correct)
    recall_tue_50 = (y_pred_50[masque_tue_test] == 1).mean()
    proba_moy_tue = y_proba[masque_tue_test].mean()
    radd(f"\n[FOCUS TUÉS — SEUIL 0.50]")
    radd(f"  Tués dans le test             : {n_tue_test:,}")
    radd(f"  Recall sur les tués           : {recall_tue_50*100:.1f}%")
    radd(f"    → {int(recall_tue_50*n_tue_test)} tués détectés / {n_tue_test} tués réels")
    radd(f"    → {int((1-recall_tue_50)*n_tue_test)} tués RATÉS (faux négatifs critiques)")
    radd(f"  Probabilité moyenne des tués  : {proba_moy_tue:.4f}")
    radd(f"    (plus proche de 1.0 = modèle plus confiant sur les tués)")
else:
    radd(f"\n  ⚠ Aucun tué dans le jeu de test (échantillon trop petit ?)")

# =============================================================================
# PHASE 5 — OPTIMISATION DU SEUIL (recall Tués)
# =============================================================================
# On abaisse le seuil pour maximiser le recall spécifiquement sur les TUÉS.
# Contrainte : recall_tués ≥ TARGET_RECALL_TUE (75%) ET precision_globale ≥ TARGET_PREC
#
# MÉCANIQUE :
#   Seuil 0.50 → si proba ≥ 0.50 → prédit "reste" (classe 1)
#   Seuil 0.30 → si proba ≥ 0.30 → prédit "reste"
#   → En abaissant le seuil, on prédit "reste" plus souvent
#   → On capture plus de tués (recall augmente) mais plus de fausses alarmes
print(f"\n[PHASE 5] Optimisation du seuil de décision...")
print(f"  Objectif : recall_tués ≥ {TARGET_RECALL_TUE:.0%}  ET  precision_classe1 ≥ {TARGET_PREC_GLOBALE:.0%}")

# Calcul du recall tués et de la precision classe 1 pour chaque seuil possible
seuils = np.linspace(0.05, 0.95, 200)
recalls_tue   = []
precisions_g1 = []
f1_list       = []

for seuil in seuils:
    y_pred_seuil = (y_proba >= seuil).astype(int)
    # Recall tués = parmi les vrais tués, combien prédits "classe 1" ?
    if masque_tue_test.sum() > 0:
        r_tue = (y_pred_seuil[masque_tue_test] == 1).mean()
    else:
        r_tue = 0.0
    # Precision classe 1 globale
    p_g1 = precision_score(y_test, y_pred_seuil, pos_label=1, zero_division=0)
    f1   = f1_score(y_test, y_pred_seuil, pos_label=1, zero_division=0)
    recalls_tue.append(r_tue)
    precisions_g1.append(p_g1)
    f1_list.append(f1)

recalls_tue   = np.array(recalls_tue)
precisions_g1 = np.array(precisions_g1)
f1_list       = np.array(f1_list)

# Seuils satisfaisant les deux objectifs
candidats = (recalls_tue >= TARGET_RECALL_TUE) & (precisions_g1 >= TARGET_PREC_GLOBALE)

if candidats.sum() > 0:
    # Parmi les seuils valides, on choisit celui avec le meilleur F1 classe 1
    idx_opt         = np.where(candidats)[0][np.argmax(f1_list[candidats])]
    optimal_seuil   = float(seuils[idx_opt])
    recall_tue_opt  = recalls_tue[idx_opt]
    prec_opt        = precisions_g1[idx_opt]
    print(f"  ✓ Seuil optimal trouvé : {optimal_seuil:.4f}")
    print(f"    Recall tués          : {recall_tue_opt*100:.1f}%  (objectif ≥ {TARGET_RECALL_TUE*100:.0f}%)")
    print(f"    Precision classe 1   : {prec_opt*100:.1f}%  (objectif ≥ {TARGET_PREC_GLOBALE*100:.0f}%)")
else:
    # Aucun seuil ne satisfait les deux → on maximise le recall tués seul
    idx_opt         = np.argmax(recalls_tue)
    optimal_seuil   = float(seuils[idx_opt])
    recall_tue_opt  = recalls_tue[idx_opt]
    prec_opt        = precisions_g1[idx_opt]
    print(f"  ⚠ Aucun seuil ne satisfait les deux objectifs simultanément")
    print(f"    → Seuil choisi par recall_tués maximum : {optimal_seuil:.4f}")

# Application du seuil optimal
y_pred_opt    = (y_proba >= optimal_seuil).astype(int)
acc_opt       = accuracy_score(y_test, y_pred_opt)
recall_g1_opt = recall_score(y_test, y_pred_opt, pos_label=1, zero_division=0)
prec_g1_opt   = precision_score(y_test, y_pred_opt, pos_label=1, zero_division=0)
f1_g1_opt     = f1_score(y_test, y_pred_opt, pos_label=1, zero_division=0)

if masque_tue_test.sum() > 0:
    recall_tue_final  = (y_pred_opt[masque_tue_test] == 1).mean()
    n_tue_detectes    = int(recall_tue_final * n_tue_test)
    n_tue_rates       = n_tue_test - n_tue_detectes

radd("\n[OPTIMISATION DU SEUIL DE DÉCISION]")
radd(f"  Seuil par défaut : 0.50  →  Seuil optimal : {optimal_seuil:.4f}")
radd(f"\n  {'':22}  {'Seuil 0.50':>11}  {'Seuil optimal':>13}")
radd(f"  {'─'*22}  {'─'*11}  {'─'*13}")
radd(f"  {'Seuil utilisé':<22}  {'0.50':>11}  {optimal_seuil:>13.4f}")
radd(f"  {'Accuracy':<22}  {acc*100:>10.1f}%  {acc_opt*100:>12.1f}%")
radd(f"  {'Recall classe 1':<22}  {recall_g1*100:>10.1f}%  {recall_g1_opt*100:>12.1f}%")
radd(f"  {'Precision classe 1':<22}  {prec_g1*100:>10.1f}%  {prec_g1_opt*100:>12.1f}%")
radd(f"  {'F1 classe 1':<22}  {f1_g1*100:>10.1f}%  {f1_g1_opt*100:>12.1f}%")
if masque_tue_test.sum() > 0:
    radd(f"  {'Recall TUÉS':<22}  {recall_tue_50*100:>10.1f}%  {recall_tue_final*100:>12.1f}%  ← FOCUS")
    radd(f"\n  → Avec le seuil optimal :")
    radd(f"    {n_tue_detectes} tués détectés sur {n_tue_test} ({recall_tue_final*100:.1f}%)")
    radd(f"    {n_tue_rates} tués encore ratés ({(1-recall_tue_final)*100:.1f}%)")

# =============================================================================
# PHASE 6 — VISUALISATIONS
# =============================================================================
print("\n[PHASE 6] Création des visualisations...")

# ── Courbe ROC ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"XGBoost (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Aléatoire (AUC = 0.5)")
ax.set_xlabel("Taux de faux positifs", fontsize=12)
ax.set_ylabel("Taux de vrais positifs (Recall)", fontsize=12)
ax.set_title("Courbe ROC — focus Tués\nIndemne vs Reste (focus Tués)", fontsize=12)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(ROC_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {ROC_PNG}")

# ── Courbe Precision-Recall (globale + point seuil optimal) ───────────────────
fig, ax = plt.subplots(figsize=(9, 6))
prec_curve, rec_curve, thr_curve = precision_recall_curve(y_test, y_proba)
ax.plot(rec_curve, prec_curve, color="steelblue", lw=2, label=f"XGBoost (AP = {pr_auc:.3f})")
ax.plot(recall_g1_opt, prec_g1_opt, "r*", markersize=15,
        label=f"Seuil optimal ({optimal_seuil:.2f})\nRecall={recall_g1_opt:.2f} Prec={prec_g1_opt:.2f}")
ax.axhline(y=TARGET_PREC_GLOBALE, color="orange", linestyle="--", lw=1.5,
           label=f"Cible precision ≥ {TARGET_PREC_GLOBALE:.0%}", alpha=0.7)
baseline = y_test.mean()
ax.axhline(y=baseline, color="gray", linestyle=":", lw=1.5,
           label=f"Baseline ({baseline:.2f})", alpha=0.7)
ax.set_xlabel("Recall classe 1", fontsize=12)
ax.set_ylabel("Precision classe 1", fontsize=12)
ax.set_title("Courbe Precision-Recall — focus Tués\n(★ = seuil optimal pour les Tués)", fontsize=12)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(PR_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {PR_PNG}")

# ── Importance des variables ───────────────────────────────────────────────────
try:
    importances = xgb_model.feature_importances_
    fi_df = pd.DataFrame({"Variable": X_full.columns.tolist(), "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(fi_df["Variable"], fi_df["Importance"], color="steelblue", edgecolor="white")
    ax.set_xlabel("Importance (gain)", fontsize=12)
    ax.set_title("Top 20 variables — focus Tués\n(sample_weights focus Tués)", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FI_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {FI_PNG}")

    top5_fi = fi_df.sort_values("Importance", ascending=False).head(5)
    radd("\n[IMPORTANCE DES VARIABLES — Top 5]")
    for _, row in top5_fi.iterrows():
        radd(f"  {row['Variable']:<15}  {row['Importance']:.4f}")
except Exception as e:
    print(f"  ⚠ Erreur importance variables : {e}")

# ── Matrice de confusion (seuil optimal) ──────────────────────────────────────
try:
    cm = confusion_matrix(y_test, y_pred_opt)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Indemne (0)", "Reste (1)"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Prédit", fontsize=11)
    ax.set_ylabel("Réel", fontsize=11)
    ax.set_title(f"Matrice de confusion — seuil {optimal_seuil:.2f}", fontsize=11)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=12)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {CM_PNG}")
except Exception as e:
    print(f"  ⚠ Erreur matrice de confusion : {e}")

# =============================================================================
# PHASE 7 — SAUVEGARDE FINALE
# =============================================================================
print("\n[PHASE 7] Sauvegarde finale...")

joblib.dump(xgb_model, MODEL_FILE, compress=3)
joblib.dump(optimal_seuil, THRESHOLD_FILE)
print(f"  ✓ Modèle sauvegardé     : {MODEL_FILE}")
print(f"  ✓ Seuil optimal sauvegardé : {THRESHOLD_FILE}  (valeur : {optimal_seuil:.4f})")

# Rapport texte final
radd("\n" + "=" * 68)
radd(f"  FICHIERS PRODUITS — training_focus_tues.py")
radd("=" * 68)
radd(f"  Checkpoints :")
radd(f"    Modèle intermédiaire : {CHECKPOINT_MODEL_INTERMED}")
radd(f"  Modèle final   : {MODEL_FILE}")
radd(f"  Seuil optimal  : {THRESHOLD_FILE}  ({optimal_seuil:.4f})")
radd(f"  Rapport        : {REPORT_FILE}")
radd(f"  Courbe ROC     : {ROC_PNG}")
radd(f"  Courbe PR      : {PR_PNG}")
radd(f"  Importance     : {FI_PNG}")
radd(f"  Matrice conf.  : {CM_PNG}")
radd("")
radd(f"  RÉSUMÉ PERFORMANCE :")
radd(f"    Seuil par défaut (0.50) :")
radd(f"      Recall classe 1     : {recall_g1*100:.1f}%")
if masque_tue_test.sum() > 0:
    radd(f"      Recall Tués         : {recall_tue_50*100:.1f}%")
radd(f"    Seuil optimal ({optimal_seuil:.4f}) :")
radd(f"      Recall classe 1     : {recall_g1_opt*100:.1f}%")
if masque_tue_test.sum() > 0:
    radd(f"      Recall Tués         : {recall_tue_final*100:.1f}%  ← OBJECTIF")
    radd(f"      Tués détectés       : {n_tue_detectes}/{n_tue_test}")
radd("=" * 68)

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"  ✓ Rapport sauvegardé    : {REPORT_FILE}")

print("\n" + "=" * 70)
print("  ✓ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
print("=" * 70)

# ── Utilisation en production ──────────────────────────────────────────────────
print("""
  POUR PRÉDIRE SUR DE NOUVELLES DONNÉES :
    import joblib
    model     = joblib.load("model_focus_tues.pkl")
    threshold = joblib.load("threshold_focus_tues.pkl")
    probas    = model.predict_proba(X_new)[:, 1]
    preds     = (probas >= threshold).astype(int)
    # preds=0 → Indemne | preds=1 → Non-indemne (Tué/Hospitalisé/Blessé)
""")
