# -*- coding: utf-8 -*-
"""
Tests de l'API FastAPI.

On utilise TestClient (fourni par FastAPI) pour simuler des requetes HTTP
SANS lancer de vrai serveur.

Les fonctions train_model() et predict_model() sont remplacees par de fausses
versions qui renvoient un resultat fixe. Ca permet de tester la logique
de l'API sans infrastructure (pas de PostgreSQL, pas de modele .pkl).
"""

import sys
import os
from unittest.mock import MagicMock

# ============================================================================
# Ajouter la racine du projet au PYTHONPATH
# pour que "from src.models.api import api" fonctionne
# ============================================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ============================================================================
# Mocker training_v2 et predict_v2 AVANT que api.py les importe
# On cree de faux modules qui renvoient des resultats fixes
# ============================================================================
mock_training = MagicMock()
mock_predict = MagicMock()

mock_training.train_model.return_value = {
    "status": "training completed",
    "accuracy": 0.81,
    "f1_macro": 0.79,
}
mock_predict.predict_model.return_value = {
    "status": "prediction completed",
    "n_predictions": 100,
    "n_indemne": 40,
    "n_reste": 60,
}

# Injecter les faux modules AVANT l'import de api.py
sys.modules["src.models.training_v2"] = mock_training
sys.modules["src.models.predict_v2"] = mock_predict

# Maintenant on peut importer api.py sans erreur
from src.models.api import api  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# Client de test FastAPI
client = TestClient(api)


class TestTrainingEndpoint:
    """Tests de l'endpoint POST /training/."""

    def test_training_returns_200(self):
        """L'endpoint /training/ doit renvoyer 200 quand tout va bien."""
        response = client.post("/training/")
        assert response.status_code == 200

    def test_training_returns_json(self):
        """La reponse doit contenir les cles attendues."""
        response = client.post("/training/")
        data = response.json()

        assert "status" in data
        assert data["status"] == "training completed"
        assert "accuracy" in data
        assert "f1_macro" in data

    def test_training_error_returns_500(self):
        """Si train_model() leve une exception, l'API doit renvoyer 500."""
        mock_training.train_model.side_effect = Exception("DB connection failed")

        response = client.post("/training/")
        assert response.status_code == 500

        # Restaurer le comportement normal apres le test
        mock_training.train_model.side_effect = None
        mock_training.train_model.return_value = {
            "status": "training completed",
            "accuracy": 0.81,
            "f1_macro": 0.79,
        }


class TestPredictEndpoint:
    """Tests de l'endpoint POST /predict/."""

    def test_predict_returns_200(self):
        """L'endpoint /predict/ doit renvoyer 200."""
        response = client.post("/predict/")
        assert response.status_code == 200

    def test_predict_returns_json(self):
        """La reponse doit contenir les cles de prediction."""
        response = client.post("/predict/")
        data = response.json()

        assert data["status"] == "prediction completed"
        assert "n_predictions" in data
        assert "n_indemne" in data
        assert "n_reste" in data

    def test_predict_counts_coherent(self):
        """n_indemne + n_reste doit egal n_predictions."""
        response = client.post("/predict/")
        data = response.json()

        assert data["n_indemne"] + data["n_reste"] == data["n_predictions"]


class TestMethodesNonAutorisees:
    """Tests que les mauvaises methodes HTTP sont rejetees."""

    def test_get_training_not_allowed(self):
        """GET /training/ ne doit pas etre autorise (seul POST l'est)."""
        response = client.get("/training/")
        assert response.status_code == 405  # Method Not Allowed

    def test_get_predict_not_allowed(self):
        """GET /predict/ ne doit pas etre autorise."""
        response = client.get("/predict/")
        assert response.status_code == 405
