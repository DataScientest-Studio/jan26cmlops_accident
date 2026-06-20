# 🚗 MLOps Accidentologie — Prédiction de la gravité des accidents de la route

Projet MLOps de bout en bout — DataScientest Janvier 2026  
**Équipe** : Chaymae Gasmi, Sébastien Mével, Marc Guezou | **Mentor** : Nicolas

---

## Objectif

Prédire si un usager impliqué dans un accident de la route sera **indemne ou blessé/tué**, avec un focus sur la détection des accidents mortels (classe rare < 2%).

---

## Lancement rapide

```bash
# Premier lancement (charge les données)
docker compose --profile init up --build

# Lancements suivants
docker compose up
```

---

## Services

| Service    | URL                          | Description            |
|------------|------------------------------|------------------------|
| API        | http://localhost:8000/docs   | Documentation Swagger  |
| MLflow     | http://localhost:8080        | Suivi des expériences  |
| Streamlit  | http://localhost:8501        | Interface présentation |
| PostgreSQL | localhost:5432               | Base de données        |

---

## Organisation du projet
```
jan26cmlops_accident/
│
├── README.md                   <- Ce fichier
├── .env                        <- Variables d'environnement (non versionné)
├── .gitignore
│
├── Dockerfile                  <- Conteneur API FastAPI et db-init
├── Dockerfile.streamlit        <- Conteneur Streamlit
├── docker-compose.yml          <- Orchestration des 5 services Docker
│
├── requirements.docker.txt     <- Dépendances Python pour Docker (Linux)
├── requirements.streamlit.txt  <- Dépendances Streamlit
├── requirements.txt            <- Dépendances Python complètes
│
├── streamlit_app.py            <- Application Streamlit (présentation)
│
├── data_kaggle/                <- CSV BAAC (non versionnés sur Git)
│   ├── CARACTERISTICS.csv      <- Conditions de l'accident (date, météo, lieu...)
│   ├── PLACES.csv              <- Type de route, infrastructure...
│   ├── VEHICLES.csv            <- Informations sur les véhicules
│   ├── USERS.csv               <- Usagers impliqués et gravité
│   └── HOLIDAYS.csv            <- Jours fériés et vacances scolaires
│
├── models/                     <- Modèles entraînés
│   ├── model_focus_tues.pkl    <- Modèle XGBoost en production
│   └── threshold_focus_tues.pkl <- Seuil optimal de décision (0.66)
│
├── reports/                    <- Rapports générés
│   ├── figures/                <- Graphiques (ROC, PR, Feature Importance...)
│   ├── data_drift.html         <- Rapport Evidently data drift
│   └── model_performance.html  <- Rapport Evidently performance
│
├── src/
│   ├── data/
│   │   └── database/
│   │       ├── create_table.sql    <- Création des tables PostgreSQL
│   │       ├── init_mlflow_db.sql  <- Création base mlflow_tracking
│   │       └── fill_database.py    <- Chargement des CSV dans PostgreSQL
│   │
│   └── models/
│       ├── api.py              <- API FastAPI (tous les endpoints)
│       ├── training_v2.py      <- Pipeline d'entraînement XGBoost (9 phases)
│       ├── predict_v2.py       <- Script de prédiction
│       └── monitoring.py       <- Monitoring Evidently (data drift)
│
├── tests/                      <- Tests unitaires (pytest)
│
└── .github/
    └── workflows/
        └── python-app.yml      <- CI/CD GitHub Actions (lint, tests, build)
```
---
---

## Stack technique

| Composant     | Technologie                                        |
|---------------|----------------------------------------------------|
| Modèle        | XGBoost binaire — AUC = 0.8991 — seuil = 0.66     |
| Base données  | PostgreSQL 15                                      |
| Tracking ML   | MLflow avec backend PostgreSQL                     |
| Versionning   | DVC — hash MD5 loggués dans MLflow                 |
| Orchestration | Docker Compose — 5 services                        |
| CI/CD         | GitHub Actions — lint + tests + build              |
| Monitoring    | Evidently — retraining auto si drift > 30%         |
| Interface     | Streamlit                                          |

---

<p><small>Projet MLOps — DataScientest 2026</small></p>
