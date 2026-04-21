# -*- coding: utf-8 -*-
"""
=============================================================================
  preprocess_focus_tues.py — Preprocessing amélioré (basé sur Seb + final)
=============================================================================

OBJECTIF : Transformer les données brutes PostgreSQL en dataset prêt pour ML.

NOUVELLE CIBLE (vs preprocess_final.py) :
  grav_bin = 0 : Indemne    (grav=1)
  grav_bin = 1 : Reste      (grav=2 Tué + grav=3 Hospitalisé + grav=4 Blessé léger)

  POURQUOI ce choix ?
  → Le modèle de training utilisera des sample_weights pour donner un poids
    supplémentaire aux "Tués" (grav=2) lors de l'entraînement.
    La colonne 'grav' est donc CONSERVÉE dans le dataset final.

AMÉLIORATIONS vs Seb (notebook) :
  - Connexion via SQLAlchemy (meilleure compatibilité pandas)
  - Rapport d'anomalies AVANT nettoyage (trace écrite)
  - GPS 0 → NaN (lignes conservées, pas supprimées)
  - Feature engineering : age, heure, is_weekend, is_holiday
  - Valeurs -1 conservées comme catégorie valide
  - Checkpoints Parquet à chaque phase

AMÉLIORATIONS vs preprocess_final.py :
  - Nouvelle définition de grav_bin (indemne vs TOUT le reste)
  - Colonne 'grav' conservée pour les sample_weights en training
  - Checkpoint Phase 2 (avant nettoyage, pour audit)

PHASES :
  PHASE 1 : Chargement depuis PostgreSQL (jointure 4 tables)
  PHASE 2 : Rapport d'anomalies + checkpoint AVANT nettoyage
  PHASE 3 : Nettoyage + feature engineering
  PHASE 4 : Création de grav_bin + bilan + sauvegarde finale

Usage (depuis la racine du repo) :
  python src/data/preprocess_focus_tues.py

Auteur : Projet MLOps Accidents — DataScientest
Date   : Avril 2026
=============================================================================
"""

import os
import sys
import io
import time
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import pandas as pd
from sqlalchemy import create_engine, text

# =============================================================================
# CONFIGURATION
# =============================================================================

DB_HOST     = "localhost"
DB_PORT     = "5432"
DB_NAME     = "accidents"
DB_USER     = "postgres"
DB_PASSWORD = "1234"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Racine du repo = 2 niveaux au-dessus de src/data/
REPO_ROOT          = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR           = os.path.join(REPO_ROOT, "src", "data")
REPORTS_DIR        = os.path.join(REPO_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

REPORT_FILE        = os.path.join(REPORTS_DIR, "anomalies_report_focus_tues.txt")
CHECKPOINT_RAW     = os.path.join(DATA_DIR, "checkpoint_01_raw.parquet")             # Après jointure SQL
CHECKPOINT_BEFORE  = os.path.join(DATA_DIR, "checkpoint_02_before_clean.parquet")    # Avant nettoyage (audit)
PARQUET_FINAL      = os.path.join(DATA_DIR, "dataset_focus_tues.parquet")            # Dataset final

# Colonnes catégorielles utilisant -1 pour "non renseigné"
COLS_NR = [
    "lum", "agg", "int", "atm", "col",
    "catr", "circ", "nbv", "vosp", "prof", "plan", "surf", "infra", "situ", "env1",
    "senc", "obs", "obsm", "choc", "manv",
    "place", "trajet", "secu", "locp", "actp", "etatp",
]

print("=" * 70)
print("  PREPROCESSING focus_tues — Indemne vs Reste (focus Tués)")
print("=" * 70)

lines = []

def add(line=""):
    lines.append(line)
    print(line)

# =============================================================================
# PHASE 1 — CHARGEMENT DEPUIS POSTGRESQL
# =============================================================================
# Même jointure que preprocess_final.py (4 tables : users + caracté + places + vehicles)
# + LEFT JOIN holidays pour la variable is_holiday.
print("\n[PHASE 1] Chargement depuis PostgreSQL...")

t0 = time.time()
try:
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("  ✓ Connexion à la base de données OK")
except Exception as e:
    print(f"  ✗ Erreur de connexion : {e}")
    print("    Vérifiez que PostgreSQL est démarré et que les paramètres sont corrects.")
    sys.exit(1)

SQL_JOIN = """
    SELECT
        u.id_user,
        c.num_acc,
        u.num_veh,

        -- Contexte de l'accident (caracteristics)
        c.an, c.mois, c.jour, c.hrmn,
        c.lum, c.agg, c."int", c.atm, c.col,
        c.com, c.lat, c.long, c.dep,

        -- Caractéristiques de la route (places)
        p.catr, p.circ, p.nbv, p.vosp, p.prof, p.plan,
        p.lartpc, p.larrout,
        p.surf, p.infra, p.situ, p.env1,

        -- Véhicule (vehicles)
        v.senc, v.catv, v.occutc,
        v.obs, v.obsm, v.choc, v.manv,

        -- Usager (users)
        u.place, u.catu,
        u.grav,       -- CONSERVÉ pour les sample_weights dans training
        u.sexe, u.trajet, u.secu,
        u.locp, u.actp, u.etatp,
        u.an_nais,

        -- Jour férié (holidays)
        CASE WHEN h.ds IS NOT NULL THEN 1 ELSE 0 END AS is_holiday

    FROM users u
        JOIN caracteristics c ON u.num_acc = c.num_acc
        JOIN places p         ON u.num_acc = p.num_acc
        JOIN vehicles v       ON u.num_acc = v.num_acc AND u.num_veh = v.num_veh
        LEFT JOIN holidays h  ON MAKE_DATE(2000 + c.an, c.mois, c.jour) = h.ds
"""

print("  → Exécution de la jointure 4 tables...", flush=True)
df = pd.read_sql(SQL_JOIN, engine)
elapsed = time.time() - t0
print(f"  ✓ Dataset chargé : {len(df):,} lignes × {df.shape[1]} colonnes ({elapsed:.1f}s)")

# CHECKPOINT 1 — données brutes après jointure
print(f"\n  → Checkpoint 1 : sauvegarde brute avant nettoyage...")
df.to_parquet(CHECKPOINT_RAW, index=False, compression="snappy")
print(f"  ✓ {CHECKPOINT_RAW}")

# =============================================================================
# PHASE 2 — RAPPORT D'ANOMALIES AVANT NETTOYAGE
# =============================================================================
print("\n[PHASE 2] Rapport d'anomalies AVANT nettoyage...")

n_total = len(df)

add("=" * 68)
add(f"  RAPPORT D'ANOMALIES — focus_tues")
add(f"  Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
add(f"  Lignes totales (après jointure) : {n_total:,}")
add("=" * 68)

# ── 2a. Doublons ──────────────────────────────────────────────────────────────
add("\n[2a. DOUBLONS]")
n_dup_strict = df.duplicated(subset=["id_user"]).sum()
add(f"  Doublons stricts (id_user)       : {n_dup_strict:>10,}  ({n_dup_strict/n_total*100:.2f}%)")

dup_cols = ["num_acc", "num_veh", "place", "catu", "sexe", "grav"]
n_dup_fonc = df.duplicated(subset=dup_cols).sum()
add(f"  Doublons fonctionnels            : {n_dup_fonc:>10,}  ({n_dup_fonc/n_total*100:.2f}%) → à supprimer")

# ── 2b. Variable cible grav ───────────────────────────────────────────────────
add("\n[2b. VARIABLE CIBLE — grav]")
add("    1=Indemne  2=Tué  3=Hospitalisé  4=Blessé léger")
add("")
add("    NOUVELLE CIBLE grav_bin :")
add("      0 = Indemne (grav=1)")
add("      1 = Reste   (grav=2 Tué + grav=3 Hospitalisé + grav=4 Blessé léger)")
add("")

grav_labels = {1: "Indemne", 2: "Tué", 3: "Hospitalisé", 4: "Blessé léger"}
grav_counts = df["grav"].value_counts().sort_index()
for g, label in grav_labels.items():
    n = grav_counts.get(g, 0)
    add(f"  grav={g}  {label:<15} : {n:>10,}  ({n/n_total*100:.2f}%)")

n_grav_ko = (~df["grav"].isin([1, 2, 3, 4])).sum()
if n_grav_ko > 0:
    add(f"  grav invalide (hors 1-4)         : {n_grav_ko:>10,} → à supprimer")

n_indemne   = grav_counts.get(1, 0)
n_reste     = n_total - n_indemne
add(f"\n  grav_bin=0 (indemne)             : {n_indemne:>10,}  ({n_indemne/n_total*100:.2f}%)")
add(f"  grav_bin=1 (reste)               : {n_reste:>10,}  ({n_reste/n_total*100:.2f}%)")
add(f"  Ratio reste/indemne              : {n_reste/n_indemne:.2f}x")

# ── 2c. Valeurs manquantes ─────────────────────────────────────────────────────
add("\n[2c. VALEURS MANQUANTES (NULL)]")
cols_nullable = {
    "an_nais" : "Année de naissance",
    "lat"     : "Latitude GPS",
    "long"    : "Longitude GPS",
    "lartpc"  : "Largeur terre-plein central",
    "larrout" : "Largeur chaussée",
}
for col, label in cols_nullable.items():
    if col in df.columns:
        n_null = df[col].isna().sum()
        add(f"  {col:<12} ({label:<28}) : {n_null:>10,}  ({n_null/n_total*100:.2f}%)")

n_lat0 = ((df["lat"] == 0) | df["lat"].isna()).sum()
add(f"\n  GPS manquant (lat=0 ou NULL)     : {n_lat0:>10,}  ({n_lat0/n_total*100:.2f}%)")
add(f"    → STRATÉGIE : 0 → NaN, lignes CONSERVÉES (73% sans GPS dans ce dataset)")

# ── 2d. Valeurs -1 (non renseigné) ────────────────────────────────────────────
add("\n[2d. VALEURS -1 (non renseigné) → CONSERVÉES comme catégorie valide]")
add(f"  {'Colonne':<10}  {'Nb de -1':>10}  {'Taux':>8}")
add(f"  {'-'*10}  {'-'*10}  {'-'*8}")
for col in COLS_NR:
    if col in df.columns:
        n_nr = (df[col] == -1).sum()
        pct  = n_nr / n_total * 100
        flag = "  ⚠ >50%" if pct > 50 else ("  ! >30%" if pct > 30 else "")
        add(f"  {col:<10}  {n_nr:>10,}  {pct:>7.1f}%{flag}")

# ── 2e. Valeurs aberrantes ────────────────────────────────────────────────────
add("\n[2e. VALEURS ABERRANTES]")
n_sexe_ko   = (~df["sexe"].isin([1, 2])).sum()
n_catu_ko   = (~df["catu"].isin([1, 2, 3])).sum()
df["_age"]  = (2000 + df["an"]) - df["an_nais"]
n_age_ko    = ((df["_age"] < 0) | (df["_age"] > 120)).sum()
hh = df["hrmn"] // 100
mm = df["hrmn"] % 100
n_hrmn_ko = ((hh > 23) | (mm > 59) | (df["hrmn"] < 0)).sum()

add(f"  grav hors {{1,2,3,4}}             : {n_grav_ko:>10,}  ({n_grav_ko/n_total*100:.2f}%)")
add(f"  sexe hors {{1,2}}                 : {n_sexe_ko:>10,}  ({n_sexe_ko/n_total*100:.2f}%)")
add(f"  catu hors {{1,2,3}}               : {n_catu_ko:>10,}  ({n_catu_ko/n_total*100:.2f}%)")
add(f"  âge < 0 ou > 120 ans            : {n_age_ko:>10,}  ({n_age_ko/n_total*100:.2f}%)")
add(f"  hrmn invalide (HH>23/MM>59)     : {n_hrmn_ko:>10,}  ({n_hrmn_ko/n_total*100:.2f}%) → corrigé en -1")
df.drop(columns=["_age"], inplace=True, errors="ignore")

add("\n" + "=" * 68)

# Sauvegarde rapport Phase 2
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\n  Rapport Phase 2 sauvegardé : {REPORT_FILE}")

# CHECKPOINT 2 — avant nettoyage (identique au brut, utile pour audit/comparaison)
df.to_parquet(CHECKPOINT_BEFORE, index=False, compression="snappy")
print(f"  ✓ Checkpoint 2 (avant nettoyage) : {CHECKPOINT_BEFORE}")

# =============================================================================
# PHASE 3 — NETTOYAGE + FEATURE ENGINEERING
# =============================================================================
print("\n[PHASE 3] Nettoyage et feature engineering...")

n_avant = len(df)

# ── 3a. Doublons fonctionnels ──────────────────────────────────────────────────
print(f"\n  [3a] Doublons fonctionnels — avant : {len(df):,}")
n_ref = len(df)
df = df.drop_duplicates(subset=dup_cols, keep="first")
print(f"       Après : {len(df):,}  (supprimé : {n_ref - len(df):,})")

# ── 3b. Valeurs invalides grav, sexe, catu ────────────────────────────────────
print(f"\n  [3b] Valeurs invalides")
for colonne, valides in [("grav", [1,2,3,4]), ("sexe", [1,2]), ("catu", [1,2,3])]:
    n_ref = len(df)
    df = df[df[colonne].isin(valides)]
    print(f"       {colonne} invalide — supprimé : {n_ref - len(df):,}")

# ── 3c. Âge aberrant ──────────────────────────────────────────────────────────
print(f"\n  [3c] Âge aberrant")
df["_age_tmp"] = (2000 + df["an"]) - df["an_nais"]
n_ref = len(df)
masque_age_ko = df["an_nais"].notna() & ((df["_age_tmp"] < 0) | (df["_age_tmp"] > 120))
df = df[~masque_age_ko]
df.drop(columns=["_age_tmp"], inplace=True)
print(f"       Supprimé : {n_ref - len(df):,}")

# ── 3d. an_nais NULL ──────────────────────────────────────────────────────────
print(f"\n  [3d] an_nais NULL — avant : {len(df):,}")
n_ref = len(df)
df = df[df["an_nais"].notna()]
print(f"       Après : {len(df):,}  (supprimé : {n_ref - len(df):,})")

# ── 3e. GPS 0 → NaN (lignes CONSERVÉES) ──────────────────────────────────────
print(f"\n  [3e] GPS manquant : 0 → NaN (lignes conservées)")
n_lat_zero  = (df["lat"]  == 0).sum()
n_long_zero = (df["long"] == 0).sum()
df.loc[df["lat"]  == 0, "lat"]  = None
df.loc[df["long"] == 0, "long"] = None
print(f"       lat=0 → NaN  : {n_lat_zero:,} valeurs converties")
print(f"       long=0 → NaN : {n_long_zero:,} valeurs converties")
n_avant_lartpc = len(df)

# ── 3f. lartpc / larrout négatifs → NaN puis suppression ──────────────────────
print(f"\n  [3f] lartpc / larrout")
df.loc[df["lartpc"]  < 0, "lartpc"]  = None
df.loc[df["larrout"] < 0, "larrout"] = None
n_ref = len(df)
df = df[df["lartpc"].notna()]
print(f"       lartpc NaN supprimé  : {n_ref - len(df):,}")
n_ref = len(df)
df = df[df["larrout"].notna()]
print(f"       larrout NaN supprimé : {n_ref - len(df):,}")

# ── 3g. hrmn invalide → -1 (correction, pas suppression) ─────────────────────
masque_hrmn = ((df["hrmn"] // 100 > 23) | (df["hrmn"] % 100 > 59) | (df["hrmn"] < 0))
df.loc[masque_hrmn, "hrmn"] = -1
if masque_hrmn.sum() > 0:
    print(f"\n  [3g] hrmn invalide → -1 : {masque_hrmn.sum():,} valeurs corrigées")

# ── 3h. Feature engineering ────────────────────────────────────────────────────
print(f"\n  [3h] Feature engineering")

# âge (calculé depuis an_nais)
df["age"] = ((2000 + df["an"]) - df["an_nais"]).round(0).astype(int)
print(f"     age       : min={df['age'].min()}, max={df['age'].max()}, médiane={df['age'].median():.0f}")

# heure de l'accident (extrait de hrmn)
df["heure"] = df["hrmn"].apply(lambda x: x // 100 if x >= 0 else -1)
print(f"     heure     : extrait de hrmn")

# is_weekend (1 si samedi ou dimanche)
try:
    dates = pd.to_datetime(
        df["an"].apply(lambda x: 2000 + x).astype(str) + "-" +
        df["mois"].astype(str).str.zfill(2) + "-" +
        df["jour"].astype(str).str.zfill(2),
        errors="coerce"
    )
    df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    df.loc[dates.isna(), "is_weekend"] = -1
    print(f"     is_weekend: {(df['is_weekend']==1).sum():,} weekends, {(df['is_weekend']==0).sum():,} semaines")
except Exception as e:
    df["is_weekend"] = -1
    print(f"     is_weekend: erreur ({e}), mis à -1")

print(f"     is_holiday: déjà calculé via SQL")

# ── 3i. Suppression des colonnes identifiants et sources ──────────────────────
print(f"\n  [3i] Suppression des colonnes identifiants / sources")
cols_drop = [c for c in ["id_user", "num_acc", "num_veh", "an_nais", "hrmn", "an", "mois", "jour"]
             if c in df.columns]
df = df.drop(columns=cols_drop)
print(f"     Supprimées : {cols_drop}")
print(f"     Colonnes conservées : {df.shape[1]}")

# =============================================================================
# PHASE 4 — CRÉATION grav_bin + BILAN + SAUVEGARDE
# =============================================================================
print("\n[PHASE 4] Création de grav_bin + sauvegarde finale...")

# Nouvelle définition de grav_bin :
#   0 = Indemne (grav=1)
#   1 = Reste   (grav=2 Tué + grav=3 Hospitalisé + grav=4 Blessé léger)
#
# DIFFÉRENCE vs preprocess_final.py :
#   Ici, Blessé léger (grav=4) est dans la classe POSITIVE (1)
#   alors que dans preprocess_final.py, il était dans la classe NÉGATIVE (0).
#
# POURQUOI ?
#   L'objectif est de détecter TOUT accident non indemne.
#   Les "tués" (grav=2) seront sur-pondérés dans le training via sample_weight.
df["grav_bin"] = (df["grav"] != 1).astype(int)

n_final     = len(df)
n_indemne   = (df["grav_bin"] == 0).sum()
n_reste     = (df["grav_bin"] == 1).sum()
n_tue       = (df["grav"] == 2).sum()
n_hospit    = (df["grav"] == 3).sum()
n_blesse    = (df["grav"] == 4).sum()

bilan = [
    "",
    "═" * 68,
    "  BILAN FINAL — preprocess_focus_tues.py",
    "═" * 68,
    f"  Lignes chargées (jointure SQL)            : {n_avant:>10,}",
    f"  Lignes finales (après nettoyage)          : {n_final:>10,}",
    f"  Taux de conservation                      : {n_final/n_avant*100:>9.1f}%",
    f"  Colonnes finales                          : {df.shape[1]:>10,}",
    "  ─" * 34,
    f"  grav_bin=0  Indemne                       : {n_indemne:>10,}  ({n_indemne/n_final*100:.1f}%)",
    f"  grav_bin=1  Reste (total)                 : {n_reste:>10,}  ({n_reste/n_final*100:.1f}%)",
    f"    dont grav=2 Tués                        : {n_tue:>10,}  ({n_tue/n_final*100:.1f}%)",
    f"    dont grav=3 Hospitalisés               : {n_hospit:>10,}  ({n_hospit/n_final*100:.1f}%)",
    f"    dont grav=4 Blessés légers             : {n_blesse:>10,}  ({n_blesse/n_final*100:.1f}%)",
    f"  Ratio reste/indemne                       : {n_reste/n_indemne:>9.2f}x",
    "  ─" * 34,
    f"  NOTE : 'grav' est conservé dans le dataset",
    f"         → utilisé comme sample_weight dans training_focus_tues.py",
    f"         (tués×8, hospitalisés×2, blessés légers×1.5, indemnes×1)",
    "═" * 68,
    f"  Fichiers produits :",
    f"    Rapport      : {REPORT_FILE}",
    f"    Checkpoint 1 : {CHECKPOINT_RAW}",
    f"    Checkpoint 2 : {CHECKPOINT_BEFORE}",
    f"    Dataset final: {PARQUET_FINAL}",
    "═" * 68,
]

for ligne in bilan:
    print(ligne)

with open(REPORT_FILE, "a", encoding="utf-8") as f:
    f.write("\n\n" + "\n".join(bilan))

# Sauvegarde dataset final
df.to_parquet(PARQUET_FINAL, index=False, compression="snappy")
print(f"\n  ✓ Dataset final sauvegardé : {PARQUET_FINAL}")
print(f"    Taille : {os.path.getsize(PARQUET_FINAL) / 1024 / 1024:.1f} MB")

print("\n" + "=" * 70)
print("  ✓ PREPROCESSING TERMINÉ AVEC SUCCÈS")
print("=" * 70)
print(f"\n  Prochaine étape : python src/models/training_focus_tues.py")
