import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import os

#  Paramètres de connexion PostgreSQL
DB_PARAMS = {
    "dbname": "mlops_accidents",
    "user": "postgres",
    "password": "admin",  
    "host": "localhost",
    "port": 5432
}

# Chemin des CSV
CSV_FOLDER = os.path.join("..", "data_kaggle")
TABLE_CSV_MAPPING = {
    "caracteristics": "CARACTERISTICS.csv",
    "places": "PLACES.csv",
    "vehicles": "VEHICLES.csv",
    "users": "USERS.csv",
    "holidays": "HOLIDAYS.csv"
}

# Colonnes à insérer pour chaque table
TABLE_COLUMNS = {
    "caracteristics": None, 
    "places": None,
    "vehicles": None,
    "users": None,
    "holidays": None
}

# Connexion PostgreSQL
conn = psycopg2.connect(**DB_PARAMS)
cursor = conn.cursor()

# Boucle sur chaque table
for table_name, csv_file in TABLE_CSV_MAPPING.items():
    csv_path = os.path.join(CSV_FOLDER, csv_file)
    print(f"[INFO] Traitement de {table_name} à partir de {csv_file}...")

    # Lecture du csv sélectionné
    df = pd.read_csv(csv_path, encoding="latin1")

    # Suppression des doublons selon les clés primaires
    if table_name == "users":
        df = df.drop_duplicates(subset=["Num_Acc", "num_veh", "place"])
    elif table_name == "vehicles":
        df = df.drop_duplicates(subset=["Num_Acc", "num_veh"])
    elif table_name == "holidays":
        df = df.drop_duplicates(subset=["ds"])
    else:
        df = df.drop_duplicates(subset=[df.columns[0]])  # première colonne = PK

    cursor.execute(f"TRUNCATE TABLE {table_name} CASCADE;")
    conn.commit()

    # Filtrage spécifique pour users : garder que les lignes dont les véhicules existent (pb chargement database sinon)
    if table_name == "users":
        cursor.execute("SELECT Num_Acc, num_veh FROM vehicles;")
        vehicles_existing = cursor.fetchall()
        vehicles_set = set((str(v[0]), str(v[1])) for v in vehicles_existing)
        df = df[df.apply(lambda row: (str(row["Num_Acc"]), str(row["num_veh"])) in vehicles_set, axis=1)]


    cols = TABLE_COLUMNS[table_name] if TABLE_COLUMNS[table_name] else list(df.columns)
    values = [tuple(x) for x in df[cols].to_numpy()]

    # Si conflit lors du chargement (clé primaire identique), on zappe la ligne
    pk_conflict = ""
    if table_name == "users":
        pk_conflict = "ON CONFLICT (Num_Acc, num_veh, place) DO NOTHING"
    elif table_name == "vehicles":
        pk_conflict = "ON CONFLICT (Num_Acc, num_veh) DO NOTHING"
    elif table_name == "holidays":
        pk_conflict = "ON CONFLICT (ds) DO NOTHING"
    elif table_name == "places":
        pk_conflict = "ON CONFLICT (Num_Acc) DO NOTHING" 
    elif table_name == "caracteristics":
        pk_conflict = "ON CONFLICT (Num_Acc) DO NOTHING"

    # Commande SQL
    insert_query = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES %s {pk_conflict};"

    # Exécuter l'insertion
    execute_values(cursor, insert_query, values)
    conn.commit()
    print(f"[INFO] {len(values)} lignes insérées dans {table_name}.")

# Fermer connexion
cursor.close()
conn.close()
print("[INFO] Import terminé avec succès.")

