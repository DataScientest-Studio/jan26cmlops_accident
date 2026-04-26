# -*- coding: utf-8 -*-
"""
Tests du preprocessing / feature engineering.

On teste les transformations de donnees SANS base de donnees :
- Calcul de l'age a partir de an_nais et an
- Extraction de l'heure depuis hrmn
- Detection du weekend
- Encodage des "-" en -1
- GPS 0 -> NaN
"""

import numpy as np
import pandas as pd
import pytest


class TestFeatureAge:
    """Tests du calcul de l'age = an - an_nais."""

    def test_age_calcul_normal(self, sample_dataframe):
        """L'age doit etre = annee accident - annee naissance."""
        df = sample_dataframe.copy()
        df["an_nais"] = pd.to_numeric(df["an_nais"], errors="coerce")
        df["an"] = pd.to_numeric(df["an"], errors="coerce")
        df["age"] = df["an"] - df["an_nais"]

        # L'age doit etre positif et raisonnable
        assert (df["age"] >= 0).all(), "L'age ne peut pas etre negatif"
        assert (df["age"] <= 120).all(), "L'age ne peut pas depasser 120"

    def test_age_negatif_devient_nan(self):
        """Si an_nais > an, l'age doit etre NaN (donnee incoherente)."""
        df = pd.DataFrame({"an": [2020], "an_nais": [2025]})
        df["age"] = df["an"] - df["an_nais"]
        df.loc[df["age"] < 0, "age"] = np.nan

        assert df["age"].isna().sum() == 1

    def test_age_trop_grand_devient_nan(self):
        """Si age > 120, doit etre NaN."""
        df = pd.DataFrame({"an": [2020], "an_nais": [1850]})
        df["age"] = df["an"] - df["an_nais"]
        df.loc[df["age"] > 120, "age"] = np.nan

        assert df["age"].isna().sum() == 1


class TestFeatureHeure:
    """Tests de l'extraction de l'heure depuis hrmn."""

    def test_heure_extraction(self):
        """hrmn '14:30' doit donner heure = 14."""
        df = pd.DataFrame({"hrmn": ["14:30", "08:15", "23:59", "00:00"]})
        df["hrmn"] = df["hrmn"].astype(str).str.replace(":", "").str.zfill(4)
        df["heure"] = pd.to_numeric(df["hrmn"].str[:2], errors="coerce")

        assert list(df["heure"]) == [14, 8, 23, 0]

    def test_heure_bornes(self, sample_dataframe):
        """L'heure doit etre entre 0 et 23."""
        df = sample_dataframe.copy()
        df["hrmn"] = df["hrmn"].astype(str).str.replace(":", "").str.zfill(4)
        df["heure"] = pd.to_numeric(df["hrmn"].str[:2], errors="coerce")

        assert df["heure"].min() >= 0
        assert df["heure"].max() <= 23


class TestFeatureWeekend:
    """Tests de la detection du weekend."""

    def test_samedi_est_weekend(self):
        """Le samedi (dayofweek=5) doit etre is_weekend=1."""
        # 2021-01-02 = samedi
        date_tmp = pd.to_datetime("2021-01-02")
        is_weekend = int(date_tmp.dayofweek in [5, 6])

        assert is_weekend == 1

    def test_lundi_nest_pas_weekend(self):
        """Le lundi (dayofweek=0) doit etre is_weekend=0."""
        # 2021-01-04 = lundi
        date_tmp = pd.to_datetime("2021-01-04")
        is_weekend = int(date_tmp.dayofweek in [5, 6])

        assert is_weekend == 0


class TestEncodage:
    """Tests de l'encodage des valeurs speciales."""

    def test_tiret_devient_moins_un(self):
        """Les '-' doivent etre remplaces par -1."""
        df = pd.DataFrame({"col1": ["5", "-", "3", "-"]})
        df.loc[df["col1"] == "-", "col1"] = "-1"

        assert (df["col1"] == "-1").sum() == 2

    def test_gps_zero_devient_nan(self):
        """Les coordonnees GPS a 0 doivent devenir NaN."""
        df = pd.DataFrame({"lat": [48.8, 0, 45.2, 0], "long": [2.3, 0, 5.7, 0]})
        df["lat"] = df["lat"].replace(0, np.nan)
        df["long"] = df["long"].replace(0, np.nan)

        assert df["lat"].isna().sum() == 2
        assert df["long"].isna().sum() == 2


class TestCibleBinaire:
    """Tests de la creation de la cible binaire grav_bin."""

    def test_indemne_est_zero(self):
        """grav=1 (Indemne) doit donner grav_bin=0."""
        df = pd.DataFrame({"grav": [1, 2, 3, 4]})
        df["grav_bin"] = (df["grav"] != 1).astype(int)

        assert df.loc[df["grav"] == 1, "grav_bin"].iloc[0] == 0

    def test_tue_est_un(self):
        """grav=2 (Tue) doit donner grav_bin=1."""
        df = pd.DataFrame({"grav": [1, 2, 3, 4]})
        df["grav_bin"] = (df["grav"] != 1).astype(int)

        assert df.loc[df["grav"] == 2, "grav_bin"].iloc[0] == 1

    def test_distribution_binaire(self, sample_dataframe):
        """grav_bin ne doit contenir que 0 et 1."""
        df = sample_dataframe.copy()
        df["grav_bin"] = (df["grav"] != 1).astype(int)

        assert set(df["grav_bin"].unique()) == {0, 1}
