# -*- coding: utf-8 -*-
"""
Tests du modele sauvegarde.

Verifie que :
- Les fichiers modele existent
- Le modele peut etre charge par joblib
- Le modele peut faire des predictions sur des donnees factices
- Le seuil optimal est valide

Note : ces tests sont marques "skipif" si les fichiers modele n'existent pas
(cas du CI ou le training n'a pas ete lance).
"""

import os
import pytest
import numpy as np

# Chemin vers la racine du projet
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(REPO_ROOT, "models", "model_focus_tues.pkl")
THRESHOLD_FILE = os.path.join(REPO_ROOT, "models", "threshold_focus_tues.pkl")

# Condition : sauter ces tests si les fichiers n'existent pas (ex: en CI)
model_exists = pytest.mark.skipif(
    not os.path.exists(MODEL_FILE),
    reason="Modele non trouve (training pas encore lance)"
)


@model_exists
class TestModelLoading:
    """Tests de chargement du modele."""

    def test_model_file_exists(self):
        """Le fichier model_focus_tues.pkl doit exister."""
        assert os.path.exists(MODEL_FILE)

    def test_threshold_file_exists(self):
        """Le fichier threshold_focus_tues.pkl doit exister."""
        assert os.path.exists(THRESHOLD_FILE)

    def test_model_can_be_loaded(self):
        """Le modele doit pouvoir etre charge par joblib."""
        import joblib
        model = joblib.load(MODEL_FILE)
        assert model is not None

    def test_threshold_valid(self):
        """Le seuil doit etre entre 0 et 1."""
        import joblib
        meta = joblib.load(THRESHOLD_FILE)

        assert "threshold" in meta
        assert 0 < meta["threshold"] < 1

    def test_features_list_exists(self):
        """Le fichier seuil doit contenir la liste des features."""
        import joblib
        meta = joblib.load(THRESHOLD_FILE)

        assert "features" in meta
        assert len(meta["features"]) > 0

    def test_model_can_predict(self):
        """Le modele doit pouvoir predire sur des donnees factices."""
        import joblib
        model = joblib.load(MODEL_FILE)
        meta = joblib.load(THRESHOLD_FILE)

        n_features = len(meta["features"])
        # Creer une matrice de zeros avec le bon nombre de features
        X_fake = np.zeros((5, n_features))

        # Le modele doit renvoyer des probabilites
        probas = model.predict_proba(X_fake)

        assert probas.shape == (5, 2)                   # 5 lignes, 2 classes
        assert (probas >= 0).all()                       # Probas positives
        assert (probas <= 1).all()                       # Probas <= 1
        assert np.allclose(probas.sum(axis=1), 1.0)     # Somme = 1 par ligne
