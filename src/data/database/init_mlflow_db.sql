-- ============================================================================
-- init_mlflow_db.sql
-- Crée la base mlflow_tracking si elle n'existe pas.
-- MLflow a besoin de sa propre base PostgreSQL pour stocker les expériences,
-- les runs et les métriques. Sans cette base, le service MLflow crashe.
--
-- Ce script est monté dans docker-entrypoint-initdb.d/ et s'exécute
-- automatiquement au premier démarrage du conteneur PostgreSQL.
-- ============================================================================

SELECT 'CREATE DATABASE mlflow_tracking'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_tracking')\gexec
