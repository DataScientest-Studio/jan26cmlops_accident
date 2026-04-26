# -*- coding: utf-8 -*-
"""
Donnees de test preparees, partagees entre tous les fichiers de test.
conftest.py est automatiquement charge par pytest avant chaque test.
Les fonctions marquees @pytest.fixture preparent des jeux de donnees
reutilisables dans tous les tests.

============================================================================
RECAPITULATIF DES TESTS CI/CD
============================================================================
Reparer CI  : correction du workflow GitHub Actions (python-app.yml)
Ecrire tests : suite de tests automatises (ce dossier tests/)
============================================================================

test_preprocessing.py (11 tests) - Feature engineering
  TestFeatureAge:
    - test_age_calcul_normal         : age = an - an_nais, toujours positif
    - test_age_negatif_devient_nan   : an_nais > an -> NaN
    - test_age_trop_grand_devient_nan: age > 120 -> NaN
  TestFeatureHeure:
    - test_heure_extraction          : "14:30" -> 14
    - test_heure_bornes              : heure entre 0 et 23
  TestFeatureWeekend:
    - test_samedi_est_weekend        : samedi -> is_weekend=1
    - test_lundi_nest_pas_weekend    : lundi -> is_weekend=0
  TestEncodage:
    - test_tiret_devient_moins_un    : "-" -> -1
    - test_gps_zero_devient_nan      : lat/long 0 -> NaN
  TestCibleBinaire:
    - test_indemne_est_zero          : grav=1 -> grav_bin=0
    - test_tue_est_un                : grav=2 -> grav_bin=1
    - test_distribution_binaire      : grav_bin contient que 0 et 1

test_api.py (8 tests) - Endpoints FastAPI
  TestTrainingEndpoint:
    - test_training_returns_200      : POST /training/ -> 200
    - test_training_returns_json     : reponse contient accuracy, f1_macro
    - test_training_error_returns_500: exception -> 500
  TestPredictEndpoint:
    - test_predict_returns_200       : POST /predict/ -> 200
    - test_predict_returns_json      : reponse contient n_predictions
    - test_predict_counts_coherent   : indemne + reste = total
  TestMethodesNonAutorisees:
    - test_get_training_not_allowed  : GET /training/ -> 405
    - test_get_predict_not_allowed   : GET /predict/ -> 405

test_model.py (6 tests) - Modele sauvegarde (sautes si absent)
  TestModelLoading:
    - test_model_file_exists         : model_focus_tues.pkl existe
    - test_threshold_file_exists     : threshold_focus_tues.pkl existe
    - test_model_can_be_loaded       : joblib.load fonctionne
    - test_threshold_valid           : seuil entre 0 et 1
    - test_features_list_exists      : liste de features non vide
    - test_model_can_predict         : predict_proba sur donnees factices

Total : 25 tests
============================================================================
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Ajouter src/models au path Python pour pouvoir importer les modules
# (meme logique que quand on fait "cd src/models && python api.py")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "data"))


@pytest.fixture
def sample_dataframe():
    """
    Cree un petit DataFrame qui ressemble aux donnees reelles
    (jointure users + vehicles + carac + places).
    Utilise dans les tests de preprocessing.

    C'est un jeu de donnees prepare : pytest l'injecte automatiquement
    dans les tests qui ont un parametre nomme "sample_dataframe".
    """
    np.random.seed(42)
    n = 100  # 100 lignes de test (vs 1.8M en prod)

    df = pd.DataFrame({
        # Identifiants
        "num_acc": range(1, n + 1),
        "num_veh": np.random.randint(1, 5, n),

        # Caracteristiques accident
        "an": np.random.choice([2019, 2020, 2021], n),
        "mois": np.random.randint(1, 13, n),
        "jour": np.random.randint(1, 29, n),
        "hrmn": [f"{h:02d}:{m:02d}" for h, m in
                 zip(np.random.randint(0, 24, n), np.random.randint(0, 60, n))],
        "lum": np.random.randint(1, 6, n),
        "agg": np.random.randint(1, 3, n),
        "int": np.random.randint(1, 10, n),
        "atm": np.random.randint(1, 9, n),
        "col": np.random.randint(1, 8, n),

        # Usager
        "catu": np.random.randint(1, 4, n),
        "sexe": np.random.choice([1, 2], n),
        "an_nais": np.random.randint(1950, 2005, n),
        "trajet": np.random.randint(1, 6, n),
        "secu": np.random.randint(1, 10, n),

        # Gravite (cible)
        "grav": np.random.choice([1, 2, 3, 4], n, p=[0.4, 0.03, 0.2, 0.37]),

        # GPS
        "lat": np.random.uniform(42, 51, n),
        "long": np.random.uniform(-5, 10, n),
    })

    return df


@pytest.fixture
def sample_features():
    """
    Liste de features attendues par le modele.
    Sous-ensemble simplifie pour les tests.
    """
    return ["lum", "agg", "int", "atm", "col", "catu", "sexe",
            "trajet", "secu", "age", "heure", "is_weekend"]
