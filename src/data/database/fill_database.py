import os
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env (racine du projet)
load_dotenv(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    ".env"
))

# Paramètres de connexion PostgreSQL
DB_PARAMS = {
    "dbname":   os.getenv("POSTGRES_DB",      "mlops_accidents"),
    "user":     os.getenv("POSTGRES_USER",     "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin"),
    "host":     os.getenv("POSTGRES_HOST",     "db"),       # ← "db" dans Docker, "localhost" en local
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
}

# Chemin des CSV — absolu, compatible Docker et local
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CSV_FOLDER = os.path.join(REPO_ROOT, "data_kaggle")        # ← chemin absolu au lieu de "../data_kaggle"

TABLE_CSV_MAPPING = {
    "caracteristics": "CARACTERISTICS.csv",
    "places":         "PLACES.csv",
    "vehicles":       "VEHICLES.csv",
    "users":          "USERS.csv",
    "holidays":       "HOLIDAYS.csv",
}

TABLE_COLUMNS = {
    "caracteristics": None,
    "places":         None,
    "vehicles":       None,
    "users":          None,
    "holidays":       None,
}

def check_already_loaded(cursor) -> bool:
    """Vérifie si les données sont déjà présentes pour éviter un rechargement inutile."""
    try:
        cursor.execute("SELECT COUNT(*) FROM caracteristics;")
        count = cursor.fetchone()[0]
        if count > 0:
            print(f"[INFO] Base déjà chargée ({count} lignes dans caracteristics). Skip.")
            return True
    except Exception:
        pass
    return False

def main():
    print("[INFO] Démarrage du chargement des données...")
    print(f"[INFO] CSV_FOLDER : {CSV_FOLDER}")
    print(f"[INFO] DB host    : {DB_PARAMS['host']}")

    # Vérifier que les CSV existent
    for table_name, csv_file in TABLE_CSV_MAPPING.items():
        csv_path = os.path.join(CSV_FOLDER, csv_file)
        if not os.path.exists(csv_path):
            print(f"[ERREUR] Fichier introuvable : {csv_path}")
            raise FileNotFoundError(f"CSV manquant : {csv_path}")

    conn   = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()

    # Skip si déjà chargé
    if check_already_loaded(cursor):
        cursor.close()
        conn.close()
        return

    # Ordre d'insertion respectant les FK : caracteristics et vehicles avant users
    ordered_tables = ["caracteristics", "places", "vehicles", "users", "holidays"]

    for table_name in ordered_tables:
        csv_file = TABLE_CSV_MAPPING[table_name]
        csv_path = os.path.join(CSV_FOLDER, csv_file)
        print(f"[INFO] Traitement de {table_name} à partir de {csv_file}...")

        df = pd.read_csv(csv_path, encoding="latin1")

        # Suppression des doublons
        if table_name == "users":
            df = df.drop_duplicates(subset=["Num_Acc", "num_veh", "place"])
        elif table_name == "vehicles":
            df = df.drop_duplicates(subset=["Num_Acc", "num_veh"])
        elif table_name == "holidays":
            df = df.drop_duplicates(subset=["ds"])
        else:
            df = df.drop_duplicates(subset=[df.columns[0]])

        cursor.execute(f"TRUNCATE TABLE {table_name} CASCADE;")
        conn.commit()

        # Filtrage users selon véhicules existants
        if table_name == "users":
            cursor.execute("SELECT Num_Acc, num_veh FROM vehicles;")
            vehicles_existing = cursor.fetchall()
            vehicles_set      = set((str(v[0]), str(v[1])) for v in vehicles_existing)
            before            = len(df)
            df = df[df.apply(
                lambda row: (str(row["Num_Acc"]), str(row["num_veh"])) in vehicles_set, axis=1
            )]
            print(f"[INFO] users : {before} → {len(df)} lignes après filtre véhicules")

        cols   = TABLE_COLUMNS[table_name] if TABLE_COLUMNS[table_name] else list(df.columns)
        values = [tuple(x) for x in df[cols].to_numpy()]

        # Clause ON CONFLICT
        pk_conflict = {
            "users":          "ON CONFLICT (Num_Acc, num_veh, place) DO NOTHING",
            "vehicles":       "ON CONFLICT (Num_Acc, num_veh) DO NOTHING",
            "holidays":       "ON CONFLICT (ds) DO NOTHING",
            "places":         "ON CONFLICT (Num_Acc) DO NOTHING",
            "caracteristics": "ON CONFLICT (Num_Acc) DO NOTHING",
        }.get(table_name, "")

        insert_query = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES %s {pk_conflict};"
        execute_values(cursor, insert_query, values)
        conn.commit()
        print(f"[INFO] {len(values)} lignes insérées dans {table_name}.")

    cursor.close()
    conn.close()
    print("[INFO] Import terminé avec succès.")


if __name__ == "__main__":
    main()