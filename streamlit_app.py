# -*- coding: utf-8 -*-
"""
=============================================================================
  streamlit_app.py - Présentation du projet MLOps Accidentologie
=============================================================================

Interface Streamlit interagissant avec l'API FastAPI pour présenter
le projet de bout en bout : contexte, architecture, ML, MLflow,
orchestration, CI/CD, monitoring et démo live.

Usage :
  streamlit run streamlit_app.py

Variable d'environnement :
  API_URL  -> URL de l'API (défaut : http://localhost:8000
              ou http://api:8000 si lancé dans Docker)
=============================================================================
"""

import os
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="MLOps Accidentologie - Présentation",
    page_icon="🚗",
    layout="wide",
)


# ============================================================================
#  HELPERS API
# ============================================================================
def api_get(path: str, timeout: int = 15):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Erreur API GET {path} : {e}")
        return None


def api_get_bytes(path: str, timeout: int = 30):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception as e:
        st.warning(f"Ressource indisponible ({path}) : {e}")
        return None


def api_post(path: str, timeout: int = 900):
    try:
        r = requests.post(f"{API_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Erreur API POST {path} : {e}")
        return None


def api_status() -> bool:
    health = api_get("/health", timeout=5)
    return health is not None and health.get("status") == "ok"


# ============================================================================
#  PAGE : INTRODUCTION
# ============================================================================
def page_intro():
    st.title("🚗 Prédiction de la gravité des accidents de la route")

    st.markdown("""
    ## Contexte

    Chaque année, les forces de l'ordre françaises enregistrent des dizaines de milliers
    d'accidents corporels via la base de données **BAAC** (Bulletin d'Analyse des Accidents
    Corporels). Ce dataset, disponible sur data.gouv.fr, couvre plusieurs années et détaille
    pour chaque accident : les usagers impliqués, les véhicules, les caractéristiques de
    l'accident (lieu, météo, luminosité...) et les lieux (type de route, infrastructure...).

    ## Objectif du projet

    Construire une chaîne **MLOps de bout en bout** capable de :
    - prédire la **gravité d'un accident** pour un usager donné,
    - tout en respectant les bonnes pratiques MLOps : reproductibilité, suivi
      d'expériences, versionning, déploiement automatisé et monitoring continu.

    ## Cible du modèle

    Le problème est reformulé en **classification binaire** :
    - `grav_bin = 0` → **Indemne**
    - `grav_bin = 1` → **Blessés/tués** (Tué, Hospitalisé ou Blessé léger)

    Ce choix permet d'appliquer des poids d'échantillonnage spécifiques pour maximiser
    le **recall sur les accidents mortels**, quitte à accepter quelques faux positifs.

    ## Roadmap du projet
    """)

    roadmap = pd.DataFrame({
        "Phase": [
            "1 - Fondations",
            "2 - Microservices, Suivi & Versionning",
            "3 - Orchestration & Déploiement",
            "4 - Monitoring & Maintenance",
            "5 - Frontend",
        ],
        "Deadline": ["1 Mars", "3 Avril", "2 Mai", "1 Juin", "20 Juin"],
        "Contenu": [
            "Base PostgreSQL, scripts training/predict, API FastAPI",
            "MLflow tracking + registry, comparaison de modèles",
            "Docker Compose, DVC, CI/CD GitHub Actions, sécurité API",
            "Evidently (drift), Prometheus/Grafana",
            "Application Streamlit + documentation",
        ],
    })
    st.table(roadmap)

    st.markdown("---")
    if api_status():
        st.success(f"✅ API connectée sur {API_URL}")
    else:
        st.error(f"❌ API non accessible sur {API_URL}")


# ============================================================================
#  PAGE : FONDATIONS
# ============================================================================
def page_fondations():
    st.title("🏗️ Fondations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Base de données PostgreSQL")
        st.markdown("""
        Les données BAAC sont chargées dans **4 tables relationnelles** :

        | Table | Contenu |
        |---|---|
        | `caracteristics` | Date, heure, conditions météo, lumière... |
        | `places` | Type de route, profil, infrastructure... |
        | `vehicles` | Catégorie de véhicule, manœuvre, choc... |
        | `users` | Usager : place, gravité (cible), âge, sexe... |

        Une table `holidays` complète le dataset avec les jours fériés français,
        utilisée pour le feature engineering (`is_holiday`).

        Le chargement est automatisé via un script `fill_database.py`, exécuté
        une seule fois au démarrage du `docker-compose` (service `db-init`).
        """)

    with col2:
        st.subheader("API FastAPI")
        st.markdown("""
        L'API expose les endpoints suivants :

        | Endpoint | Méthode | Rôle |
        |---|---|---|
        | `/training/` | POST | Lance le pipeline d'entraînement complet |
        | `/predict/` | POST | Lance les prédictions sur les données courantes |
        | `/monitoring/` | POST | Lance le monitoring Evidently (drift) |
        | `/health` | GET | Vérification de l'état de l'API |
        | `/data/stats` | GET | Statistiques du dataset |
        | `/mlflow/runs` | GET | Historique des runs MLflow |
        | `/mlflow/model` | GET | Modèle actuellement en Production |
        | `/reports/*` | GET | Rapports et figures générés |
        """)

    st.markdown("---")
    st.subheader("Aperçu des données")

    stats = api_get("/data/stats")
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Caractéristiques", f"{stats['table_counts']['caracteristics']:,}")
        c2.metric("Lieux",            f"{stats['table_counts']['places']:,}")
        c3.metric("Véhicules",        f"{stats['table_counts']['vehicles']:,}")
        c4.metric("Usagers",          f"{stats['table_counts']['users']:,}")


# ============================================================================
#  PAGE : SCRIPTS ML
# ============================================================================
def page_scripts_ml():
    st.title("🧠 Scripts ML — Pipeline d'entraînement et de prédiction")
    st.markdown("Cette section présente le pipeline de Machine Learning mis en œuvre pour la prédiction de la gravité des accidents de la route, de la préparation des données à l'évaluation du modèle.")
    st.markdown("---")
    st.subheader("📊 Distribution de la variable cible")
    st.markdown("La variable cible `grav` présente un déséquilibre de classes significatif. La modalité **Tué** (grav=2) représente moins de **2%** des observations, ce qui constitue un défi majeur pour l'apprentissage supervisé et justifie l'adoption d'une stratégie de pondération des échantillons.")
    stats = api_get("/data/stats")
    if stats:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distribution de la gravité**")
            grav_labels = stats["grav_labels"]
            grav_df = pd.DataFrame({"Gravité": [grav_labels.get(k, k) for k in stats["grav_distribution"]], "Nombre": list(stats["grav_distribution"].values())})
            st.bar_chart(grav_df.set_index("Gravité"))
        with col2:
            st.markdown("**Accidents par année**")
            year_df = pd.DataFrame({"Année": list(stats["accidents_per_year"].keys()), "Accidents": list(stats["accidents_per_year"].values())})
            st.bar_chart(year_df.set_index("Année"))
    st.markdown("---")
    st.subheader("🧹 Nettoyage & Feature Engineering")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Nettoyage des données**\n- Remplacement des coordonnées GPS nulles (valeur 0) par des valeurs manquantes (NaN)\n- Conservation des valeurs -1 comme catégorie valide représentant les données non renseignées\n- Encodage des variables catégorielles en variables numériques\n- Reformulation en classification binaire : grav_bin = 0 (Indemne) / grav_bin = 1 (Blessé ou Tué)")
    with col2:
        st.success("**Variables construites**\n- age : calculé à partir de la différence entre l'année de l'accident et l'année de naissance\n- heure : extraite de la variable hrmn\n- is_weekend : indicateur binaire (samedi ou dimanche)\n- is_holiday : indicateur binaire issu de la jointure avec la table holidays")
    st.markdown("---")
    st.subheader("🎯 Choix du modèle : XGBoost binaire focus Tués")
    col1, col2 = st.columns(2)
    with col1:
        st.warning("**Justification du choix de XGBoost**\n- Algorithme de gradient boosting adapté aux données tabulaires hétérogènes\n- Gestion native des valeurs manquantes sans imputation préalable\n- Complexité computationnelle adaptée au volume de données (1,8 million d'observations)\n- Compatibilité avec la pondération différentielle des échantillons")
    with col2:
        st.error("**Stratégie de pondération des échantillons**\n- Indemne (grav=1) : x1,0\n- Tué (grav=2) : x8,0\n- Hospitalisé (grav=3) : x2,0\n- Blessé léger (grav=4) : x1,5")
    st.markdown("> **Optimisation du seuil de décision** — Le seuil de classification par défaut (0,5) est remplacé par un seuil optimal déterminé empiriquement. Le seuil retenu **(0,62)** maximise le F1-macro sous la contrainte d'un recall sur la classe Tués supérieur ou égal à 75% et d'une précision globale supérieure ou égale à 55%.")
    st.markdown("---")
    st.subheader("📈 Résultats du dernier entraînement")
    figs = {"Courbe ROC": "roc_curve_focus_tues.png", "Precision-Recall": "pr_curve_focus_tues.png", "Feature Importance": "feature_importance_focus_tues.png", "Matrice de confusion": "confusion_matrix_focus_tues.png"}
    cols = st.columns(2)
    for i, (label, filename) in enumerate(figs.items()):
        img = api_get_bytes(f"/reports/figures/{filename}")
        with cols[i % 2]:
            st.markdown(f"**{label}**")
            if img:
                st.image(img, use_column_width=True)
            else:
                st.info("Pas encore généré — lancez un entraînement depuis la page Démo.")
    st.markdown("---")
    st.subheader("🔮 Prédiction")
    st.markdown("Le script `predict_v2.py` charge le modèle en **Production** depuis MLflow Registry et prédit sur les données courantes.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Predict sur 100k lignes", key="predict_test_page"):
            with st.spinner("Prédiction en cours..."):
                result = api_post("/predict-test/")
            if result:
                st.success("Prédiction terminée ✅")
                c1, c2, c3 = st.columns(3)
                c1.metric("Indemnes prédits", f"{result.get('n_indemne', 0):,}")
                c2.metric("Blessés/tués prédits", f"{result.get('n_non_indemne', 0):,}")
                c3.metric("Seuil utilisé", f"{result.get('threshold', 0):.2f}")
                if result.get("recall_tues") is not None:
                    st.metric("Recall Tués", f"{result['recall_tues']:.2%}")
    with col2:
        if st.button("▶️ Predict complet (1.8M)", key="predict_ml_page"):
            with st.spinner("Prédiction en cours..."):
                result = api_post("/predict/")
            if result:
                st.success("Prédiction terminée ✅")
                c1, c2, c3 = st.columns(3)
                c1.metric("Indemnes prédits", f"{result.get('n_indemne', 0):,}")
                c2.metric("Blessés/tués prédits", f"{result.get('n_non_indemne', 0):,}")
                c3.metric("Seuil utilisé", f"{result.get('threshold', 0):.2f}")
                if result.get("recall_tues") is not None:
                    st.metric("Recall Tués", f"{result['recall_tues']:.2%}")


# ============================================================================
#  PAGE : SUIVI MLFLOW
# ============================================================================
def page_mlflow():
    st.title("📡 Suivi d'expériences avec MLflow")

    st.markdown("""
    Chaque entraînement crée un **run MLflow** qui trace :
    - les **hyperparamètres** XGBoost et le seuil optimal
    - les **métriques** (accuracy, F1, AUC, recall Tués...)
    - le **modèle** XGBoost comme artefact
    - les **figures** (ROC, PR, Feature Importance, Confusion Matrix)
    - les **hash DVC** des datasets et fichiers utilisés

    Le **Model Registry** gère le cycle de vie des versions : `Staging`,
    `Production`, `Archived`. À chaque entraînement, le nouveau modèle est
    comparé au modèle en Production sur le F1-macro optimal — le meilleur
    est promu automatiquement.
    """)

    st.markdown("---")
    st.subheader("🏆 Modèle actuellement en Production")

    model_info = api_get("/mlflow/model")
    if model_info and model_info.get("status") != "no production model":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Version",      model_info["version"])
        c2.metric("F1 macro opt", f"{model_info['metrics'].get('f1_macro_opt', 0):.4f}")
        c3.metric("Recall Tués",  f"{model_info['metrics'].get('recall_tues_opt', 0):.2%}")
        c4.metric("Seuil",        f"{model_info['metrics'].get('threshold', 0):.2f}")

        with st.expander("Tous les paramètres et métriques"):
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**Paramètres**")
                st.json(model_info["params"])
            with cc2:
                st.markdown("**Métriques**")
                st.json(model_info["metrics"])
    else:
        st.info("Aucun modèle en Production pour le moment.")

    st.markdown("---")
    st.subheader("📜 Historique des runs")

    runs_data = api_get("/mlflow/runs")
    if runs_data and runs_data.get("runs"):
        runs = runs_data["runs"]

        rows = []
        for r in runs:
            rows.append({
                "Run ID":     r["run_id"][:8],
                "Type":       r["run_type"],
                "Statut":     r["status"],
                "F1 macro":   r["metrics"].get("f1_macro_opt", r["metrics"].get("f1_macro")),
                "Recall Tués": r["metrics"].get("recall_tues_opt", r["metrics"].get("recall_tues_0.5")),
                "Drift share": r["metrics"].get("drift_share"),
                "Date":       pd.to_datetime(r["start_time"], unit="ms"),
            })
        df = pd.DataFrame(rows).sort_values("Date")

        st.dataframe(df, use_container_width=True, hide_index=True)

        training_df = df[df["Type"] == "training"].dropna(subset=["F1 macro"])
        if len(training_df) > 1:
            st.markdown("**Évolution du F1-macro au fil des entraînements**")
            st.line_chart(training_df.set_index("Date")[["F1 macro", "Recall Tués"]])
    else:
        st.info("Aucun run trouvé — lancez un entraînement depuis l'onglet Démo.")

    st.markdown("---")
    st.caption(f"Interface MLflow complète : {os.getenv('MLFLOW_UI_URL', 'http://localhost:8080')}")


# ============================================================================
#  PAGE : ORCHESTRATION
# ============================================================================
def page_orchestration():
    st.title("🐳 Orchestration & Versionning")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Docker Compose")
        st.markdown("""
        L'ensemble du projet est orchestré via `docker-compose` avec
        **5 services** :

        | Service | Rôle |
        |---|---|
        | `db` | PostgreSQL — stockage des données et du backend MLflow |
        | `db-init` | Chargement initial des CSV BAAC (exécuté une fois) |
        | `mlflow` | Serveur MLflow (tracking + registry + artefacts) |
        | `api` | API FastAPI (training, predict, monitoring) |
        | `streamlit` | Cette application de présentation |

        Les services communiquent via un réseau Docker dédié (`mlops-net`),
        avec des `healthcheck` et `depends_on` pour garantir l'ordre de
        démarrage (la base doit être prête avant MLflow et l'API).
        """)

    with col2:
        st.subheader("Versionning avec DVC")
        st.markdown("""
        **DVC (sans Git)** versionne :
        - les **datasets** exportés depuis PostgreSQL (`caracteristics.csv`,
          `places.csv`, `vehicles.csv`, `users.csv`)
        - le **modèle** entraîné (`model_focus_tues.pkl`)
        - le **seuil** optimal et le **rapport** de training

        À chaque entraînement, les **hash MD5** générés par DVC sont
        loggués comme paramètres MLflow (`dvc_hash_model`,
        `dvc_hash_data_users`...) — ce qui permet de relier précisément
        une version de modèle à une version exacte des données.

        Un **remote DVC partagé** (volume Docker `dvc-storage`) permet de
        synchroniser les fichiers entre la machine hôte et les conteneurs.
        """)

    st.markdown("---")
    st.subheader("Architecture globale")
    img = api_get_bytes("/reports/figures/architecture.png")
    if img:
        st.image(img, use_column_width=True)
    else:
        st.info("Schéma d'architecture non disponible.")

# ============================================================================
#  PAGE : DÉPLOIEMENT CI/CD
# ============================================================================
def page_cicd():
    st.title("🚀 Déploiement & CI/CD")

    st.markdown("""
    Le pipeline **CI/CD** est géré par **GitHub Actions** et se déclenche
    automatiquement à chaque **push** ou **pull request** vers la branche `master`.

    Il est composé de **5 jobs** organisés en deux niveaux :
    - **4 jobs parallèles** (Tests, Sécurité, DVC, Docker) pour un feedback rapide
    - **1 job d'intégration** qui ne se lance que si les 4 premiers réussissent
    """)

    st.markdown("---")
    st.subheader("🔄 Architecture du pipeline")

    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                  Push / Pull Request → master               │
    └──────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │  Job 1       │ │  Job 2       │ │  Job 3       │
    │  Tests &     │ │  Sécurité    │ │  DVC Check   │
    │  Lint        │ │              │ │              │
    │  ≈ 57s       │ │  ≈ 8s        │ │  ≈ 15s       │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
           │       ┌────────────────┐        │
           │       │  Job 4         │        │
           │       │  Docker Build  │        │
           │       │  ≈ 90s         │        │
           │       └───────┬────────┘        │
           │               │                 │
           └───────────────┼─────────────────┘
                           │
                    needs: [1,2,3,4]
                           │
                           ▼
                ┌──────────────────┐
                │  Job 5           │
                │  Intégration     │
                │  docker compose  │
                │  ≈ 120s          │
                └──────────────────┘
    ```
    """)

    st.markdown("---")
    st.subheader("📋 Détail des 5 jobs")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1️⃣ Tests & Lint",
        "2️⃣ Sécurité",
        "3️⃣ DVC Check",
        "4️⃣ Docker Build",
        "5️⃣ Intégration",
    ])

    with tab1:
        st.markdown("#### Job 1 — Tests & Qualité du code")
        st.markdown("""
        **Objectif** : vérifier que le code est syntaxiquement correct et
        que les tests unitaires passent.

        **Étapes** :
        1. Checkout du code + setup Python 3.11
        2. Installation des dépendances (`requirements.txt` sans `pywin32` pour Linux)
        3. **Lint flake8** — détecte les erreurs critiques (imports manquants,
           variables indéfinies, syntaxe invalide)
        4. **pytest** — exécute la suite de tests avec couverture de code
        5. Sauvegarde du rapport de tests en artefact

        **Pourquoi ?** Un import manquant ou une variable indéfinie casse
        l'application en production. Flake8 bloque le merge immédiatement.
        """)
        st.code("""
# Lint bloquant (erreurs critiques uniquement)
flake8 . --select=E9,F63,F7,F82 --show-source

# Tests avec couverture
pytest tests/ -v --cov=src --cov-report=term-missing
        """, language="bash")

    with tab2:
        st.markdown("#### Job 2 — Sécurité")
        st.markdown("""
        **Objectif** : empêcher la fuite de données sensibles sur le dépôt public.

        **Contrôle 1 — Fichier `.env`** *(bloquant)* :
        - Vérifie que le fichier `.env` n'est pas commité dans le dépôt
        - Si détecté → le pipeline échoue immédiatement
        - Le `.env` contient les mots de passe PostgreSQL, les URLs de services

        **Contrôle 2 — Mots de passe en dur** *(warning)* :
        - Scanne tous les fichiers Python de `src/` avec une regex
        - Cherche les patterns `password = "valeur"` et `"password": "valeur"`
        - Exclut automatiquement les lignes contenant `os.getenv()` (pattern sécurisé)
        - Émet un warning sans bloquer le merge (évite les faux positifs)
        """)
        st.code("""
# Contrôle 1 : .env absent du repo
if [ -f .env ]; then exit 1; fi

# Contrôle 2 : scan des mots de passe en dur
grep -rn --include="*.py" \\
  -E "(password|passwd|pwd).*[:=].*['\"]" src/ \\
  | grep -v "os\\.getenv"
        """, language="bash")

    with tab3:
        st.markdown("#### Job 3 — DVC Check")
        st.markdown("""
        **Objectif** : valider la configuration DVC et éviter les erreurs de versionning.

        **Vérifications** :
        1. **`dvc doctor`** — vérifie l'installation et la config
        2. **Remote DVC** — s'assure que le remote ne pointe pas vers un
           chemin absolu Windows (`C:\\\\...`), ce qui casserait la CI Linux
        3. **Cache DVC** — vérifie que `dvc-storage/` n'est pas commité dans
           Git (c'est arrivé au début du projet : 5 Mo de données binaires)
        4. **Fichiers `.dvc`** — vérifie qu'il existe des fichiers pointeurs
           (preuve que les données sont bien suivies par DVC)

        **Contexte** : ce job a été créé suite à un incident réel où le cache
        DVC et un chemin absolu Windows avaient cassé le pipeline.
        """)

    with tab4:
        st.markdown("#### Job 4 — Docker Build")
        st.markdown("""
        **Objectif** : vérifier que l'image Docker compile et que le conteneur démarre.

        **Étapes** :
        1. Création d'un `.env` temporaire avec des valeurs de test
        2. **Validation** de la syntaxe `docker-compose.yml` (`docker compose config`)
        3. **Build** de l'image API (`docker build -t accident-api:test .`)
        4. **Smoke test** : lancement du conteneur et vérification qu'il reste
           actif pendant 5 secondes (pas de crash au démarrage)

        **Différence avec le Job 5** : ici on teste un seul conteneur isolé.
        Le job 5 teste l'ensemble des services interconnectés.
        """)

    with tab5:
        st.markdown("#### Job 5 — Intégration")
        st.markdown("""
        **Objectif** : vérifier que **tous les services fonctionnent ensemble**.

        **Prérequis** : `needs: [tests, security, dvc-check, docker-build]`
        — ne se lance que si les 4 jobs précédents sont verts.

        **Étapes** :
        1. Création d'un `.env` complet (PostgreSQL, MLflow, API)
        2. `docker compose up -d --build` — lance les 3 services
        3. Attente de 30 secondes pour le démarrage
        4. **Vérification** que `db`, `mlflow` et `api` sont en état `running`
        5. **Test HTTP** : curl sur `http://localhost:8000/docs` avec 12 tentatives
           espacées de 5 secondes (timeout total : 60s)
        6. `docker compose down -v` — nettoyage systématique

        **C'est le test le plus critique** : il valide que PostgreSQL, MLflow
        et l'API communiquent correctement via le réseau Docker `mlops-net`.
        """)

    st.markdown("---")
    st.subheader("📊 Résultats du dernier pipeline")

    results = {
        "Job": ["Tests & Lint", "Sécurité", "DVC Check", "Docker Build", "Intégration"],
        "Durée": ["~57s", "~8s", "~15s", "~90s", "~120s"],
        "Statut": ["✅ Passed", "✅ Passed", "✅ Passed", "✅ Passed", "✅ Passed"],
        "Détail": [
            "flake8 OK, pytest 100% pass",
            ".env absent, aucun password en dur",
            "Remote OK, pas de cache commité",
            "Image compile, conteneur stable",
            "3 services UP, API répond sur /docs",
        ],
    }
    st.table(pd.DataFrame(results))

    st.markdown("---")
    st.subheader("🛡️ Sécurité du projet")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Protection des secrets** :
        - Fichier `.env` exclu via `.gitignore`
        - Paramètres sensibles lus via `os.getenv()`
        - Fichier `.env.example` fourni (sans valeurs réelles)
        - CI vérifie l'absence de `.env` à chaque push
        """)
    with col2:
        st.markdown("""
        **Isolation Docker** :
        - Réseau dédié `mlops-net` entre les services
        - PostgreSQL non exposé sur l'hôte
        - Seuls les ports nécessaires sont ouverts (8000, 8080, 8501)
        - `${VAR:?Required}` empêche le démarrage sans `.env`
        """)

    st.markdown("---")
    st.subheader("🔗 Workflow Git")

    st.markdown("""
    L'équipe travaille avec un workflow **feature branch** :

    1. Chaque développeur crée une branche (`feature/xxx`)
    2. Le pipeline CI se déclenche automatiquement sur la PR
    3. Si les 5 jobs sont verts → le merge est autorisé
    4. Si un job échoue → le bouton Merge reste grisé

    **Exemple vécu** : sur la PR #4, flake8 a détecté une variable indéfinie
    dans `predict_v2.py` et le docker build a échoué par absence de `.env`.
    Correction → re-push → pipeline vert → merge autorisé.
    """)




# ============================================================================
#  PAGE : MONITORING
# ============================================================================
def page_monitoring():
    st.title("🔍 Monitoring avec Evidently")

    st.markdown("""
    Le monitoring compare deux périodes du dataset :
    - **Référence** : données historiques (années anciennes)
    - **Courant** : données récentes (dernière année du dataset = 2016)

    Deux rapports sont générés :
    1. **Data Drift** — la distribution des features a-t-elle changé ?
    2. **Performance du modèle** — le modèle se dégrade-t-il sur les données récentes ?

    Si le **taux de colonnes driftées dépasse un seuil** (30% par défaut), un
    **réentraînement est déclenché automatiquement** et le tout est tracé
    dans MLflow (run taggé `run_type=monitoring`).
    """)

    st.markdown("---")

    if st.button("▶️ Lancer le monitoring", key="monitoring_page"):
        with st.spinner("Analyse du drift en cours (peut prendre 1-2 min)..."):
            result = api_post("/monitoring/")
        if result:
            st.success("Monitoring terminé")
            c1, c2, c3 = st.columns(3)
            c1.metric("Drift share", f"{result.get('drift_share', 0):.1%}")
            c2.metric("Seuil",       f"{result.get('drift_threshold', 0):.1%}")
            c3.metric(
                "Retraining déclenché",
                "Oui ✅" if result.get("retrain_triggered") else "Non",
            )
            st.session_state["last_monitoring_result"] = result

    st.markdown("---")
    st.subheader("📄 Rapport Data Drift")

    drift_html = api_get_bytes("/reports/html/data_drift.html")
    if drift_html:
        st.components.v1.html(drift_html.decode("utf-8"), height=800, scrolling=True)
    else:
        st.info("Aucun rapport de drift disponible — lancez le monitoring ci-dessus.")

    with st.expander("📄 Rapport de performance du modèle"):
        perf_html = api_get_bytes("/reports/html/model_performance.html")
        if perf_html:
            st.components.v1.html(perf_html.decode("utf-8"), height=800, scrolling=True)
        else:
            st.info("Rapport de performance non disponible.")


# ============================================================================
#  PAGE : DÉMO
# ============================================================================
def page_demo():
    st.title("🎬 Démo en direct")

    st.markdown("""
    Cette page permet d'exécuter le pipeline complet **en direct** devant le jury :
    1. **Entraînement** — relance le pipeline et affiche les nouvelles métriques
    2. **Prédiction** — applique le modèle en Production sur les données courantes
    3. **Monitoring** — vérifie le drift et déclenche un réentraînement si besoin
    """)

    tab1, tab2, tab3 = st.tabs(["1️⃣ Training", "2️⃣ Predict", "3️⃣ Monitoring"])

    # ── Training ────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("""
        L'entraînement du modèle XGBoost est déclenché via l'endpoint `POST /training/`.
        Le pipeline complet (9 phases) est exécuté : chargement des données depuis PostgreSQL,
        nettoyage et feature engineering, sélection de features par V de Cramer, entraînement
        avec pondération des échantillons, évaluation globale, optimisation du seuil de décision,
        génération des visualisations, tracking MLflow et sauvegarde du modèle via DVC.
        L'exécution peut prendre plusieurs minutes en fonction du volume de données.
        """)
        if st.button("▶️ Lancer l'entraînement", key="demo_train"):
            with st.spinner("Entraînement en cours..."):
                result = api_post("/training/", timeout=1800)
            if result:
                st.success("Entraînement terminé")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy",     f"{result.get('accuracy', 0):.4f}")
                c2.metric("F1 macro",     f"{result.get('f1_macro', 0):.4f}")
                c3.metric("AUC-ROC",      f"{result.get('auc_roc', 0):.4f}")
                c4.metric("Recall Tués",  f"{result.get('recall_tues_opt', 0):.2%}")
                st.json(result)

        st.markdown("**Figures générées lors du dernier entraînement**")
        cols = st.columns(4)
        figs = {
            "ROC":        "roc_curve_focus_tues.png",
            "PR":         "pr_curve_focus_tues.png",
            "Features":   "feature_importance_focus_tues.png",
            "Confusion":  "confusion_matrix_focus_tues.png",
        }
        for i, (label, filename) in enumerate(figs.items()):
            img = api_get_bytes(f"/reports/figures/{filename}")
            with cols[i]:
                if img:
                    st.image(img, caption=label, use_column_width=True)

    # ── Predict ─────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        Le modèle entraîné est appliqué sur les données courantes afin de prédire
        la gravité des accidents pour chaque usager impliqué.
        Le même pipeline de prétraitement que lors de l'entraînement est appliqué
        afin de garantir la cohérence des prédictions.
        """)
        if st.button("▶️ Lancer la prédiction", key="demo_predict"):
            with st.spinner("Prédiction en cours..."):
                result = api_post("/predict/")
            if result:
                st.success("Prédiction terminée")
                c1, c2, c3 = st.columns(3)
                c1.metric("Source du modèle", result.get("model_source", "?"))
                c2.metric("Indemnes prédits",  f"{result.get('n_indemne', 0):,}")
                c3.metric("Blessés/tués prédits",     f"{result.get('n_non_indemne', 0):,}")
                if result.get("recall_tues") is not None:
                    st.metric("Recall Tués", f"{result['recall_tues']:.2%}")

                st.markdown("**Échantillon de prédictions**")
                sample = result.get("sample_predictions", [])
                if sample:
                    st.dataframe(pd.DataFrame(sample), use_container_width=True, hide_index=True)

    # ── Monitoring ──────────────────────────────────────────────────────────
    with tab3:
        st.markdown("""
        Le monitoring est réalisé via la librairie Evidently. Les données sont segmentées
        en deux périodes : les données historiques (an < 16) constituent la référence,
        et les données récentes (an ≥ 16) constituent le jeu courant. Un rapport de
        data drift et un rapport de performance du modèle sont générés automatiquement.
        Si le taux de drift dépasse le seuil de 30%, un réentraînement automatique
        est déclenché via l'endpoint `POST /training/`.
        """)
        if st.button("▶️ Lancer le monitoring", key="demo_monitoring"):
            with st.spinner("Analyse du drift en cours..."):
                result = api_post("/monitoring/")
            if result:
                st.success("Monitoring terminé")
                c1, c2, c3 = st.columns(3)
                c1.metric("Drift share", f"{result.get('drift_share', 0):.1%}")
                c2.metric("Seuil",       f"{result.get('drift_threshold', 0):.1%}")
                c3.metric(
                    "Retraining",
                    "Déclenché ✅" if result.get("retrain_triggered") else "Non déclenché",
                )

        drift_html = api_get_bytes("/reports/html/data_drift.html")
        if drift_html:
            st.markdown("**Rapport Data Drift**")
            st.components.v1.html(drift_html.decode("utf-8"), height=700, scrolling=True)


# ============================================================================
#  PAGE : ÉTAPES SUIVANTES
# ============================================================================
def page_next_steps():
    st.title("🛣️ Étapes suivantes")

    st.markdown("""
    ### Monitoring infrastructure — Prometheus & Grafana

    - Exposer les métriques de l'API (`/metrics`) via
      `prometheus-fastapi-instrumentator`
    - Dashboards Grafana pour la **latence**, le **taux d'erreurs** et le
      **nombre de requêtes** par endpoint
    - Définition d'**alertes** (ex : latence > 2s, taux d'erreur 5xx > 5%)
    - Utilisation du **webhook Grafana** pour déclencher automatiquement un
      réentraînement en cas d'anomalie détectée sur l'API

    ### Automatisation de l'entraînement

    - **Cron / Airflow** pour planifier :
      - le **monitoring Evidently** (ex : quotidien)
      - le **réentraînement** si drift détecté
      - l'**export DVC** des nouvelles données

    ### Nouvelles données

    - Le dataset BAAC est mis à jour **annuellement** sur data.gouv.fr
    - Pipeline d'ingestion incrémentale : nouvelles années → `fill_database.py`
      → nouveau `dvc add` → nouveau training automatique

    ### Kubernetes 

    - Migration de `docker-compose` vers des manifests Kubernetes
    - Scalabilité horizontale de l'API selon la charge
    - Utile si le projet doit servir plusieurs centaines de requêtes/seconde
    """)


# ============================================================================
#  PAGE : CONCLUSION
# ============================================================================
def page_conclusion():
    st.title("✅ Conclusion")
    st.markdown("""
    Ce projet a abouti à la conception et au déploiement d'une chaîne MLOps de bout en bout
    dédiée à la prédiction de la gravité des accidents de la route en France. L'objectif central
    était de construire un système capable non seulement de produire des prédictions fiables, mais
    également de garantir leur reproductibilité, leur traçabilité et leur maintien en conditions
    opérationnelles.

    Les données issues de la base BAAC, couvrant la période 2005-2016 et représentant près de
    1,8 million d'usagers impliqués dans des accidents corporels, ont été intégrées dans une base
    PostgreSQL et versionnées via DVC. Face au déséquilibre marqué de la variable cible — la classe
    Tué représentant moins de 2% des observations — un modèle XGBoost binaire a été entraîné avec
    une stratégie de pondération différentielle des échantillons, permettant d'atteindre un recall
    de 0.92 sur les accidents mortels.

    Le suivi des expériences, la gestion des versions du modèle et la promotion automatique du
    meilleur candidat en production sont assurés par MLflow. L'orchestration des cinq services
    composant l'infrastructure repose sur Docker Compose, garantissant la portabilité et la
    reproductibilité de l'environnement d'exécution. La qualité du code est contrôlée en continu
    via un workflow CI/CD GitHub Actions, tandis que la surveillance du comportement du modèle en
    production est assurée par Evidently, avec déclenchement automatique d'un réentraînement en
    cas de drift statistique supérieur à 30%.
    """)
    st.markdown("""
    | Composante | Technologie | Description |
    |---|---|---|
    | Données | PostgreSQL | Dataset BAAC — 1,8 million d'observations |
    | Modèle | XGBoost binaire | AUC-ROC = 0.79, Recall Tués = 0.92 |
    | Suivi | MLflow | Tracking, registry, comparaison de versions |
    | Versionning | DVC | Hash MD5 tracés dans MLflow |
    | Orchestration | Docker Compose | 5 services interconnectés |
    | CI/CD | GitHub Actions | Lint, tests unitaires, build |
    | Monitoring | Evidently | Data drift, retraining automatique si drift > 30% |
    | Interface | Streamlit | Présentation interactive du pipeline |
    """)



# ============================================================================
#  NAVIGATION
# ============================================================================
PAGES = {
    "📖 Introduction":         page_intro,
    "🏗️ Fondations":           page_fondations,
    "🧠 Scripts ML":           page_scripts_ml,
    "📡 Suivi MLflow":         page_mlflow,
    "🐳 Orchestration":        page_orchestration,
    "🚀 Déploiement CI/CD":    page_cicd,
    "🔍 Monitoring":           page_monitoring,
    "🎬 Démo":                 page_demo,
    "🛣️ Étapes suivantes":     page_next_steps,
    "✅ Conclusion":           page_conclusion,
}

st.sidebar.title("🚗 MLOps Accidentologie")
selection = st.sidebar.radio("Navigation", list(PAGES.keys()))

st.sidebar.markdown("---")
st.sidebar.caption(f"API : `{API_URL}`")
if api_status():
    st.sidebar.success("API connectée")
else:
    st.sidebar.error("API non accessible")

PAGES[selection]()
