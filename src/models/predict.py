# predict.py

def predict_model():

    import pandas as pd
    import psycopg2
    import joblib

    conn_params = {
        "dbname": "mlops_accidents",
        "user": "postgres",
        "password": "admin",
        "host": "localhost",
        "port": 5432
    }

    conn = psycopg2.connect(**conn_params)

    carac = pd.read_sql("SELECT * FROM caracteristics;", conn)
    places = pd.read_sql("SELECT * FROM places;", conn)
    vehicles = pd.read_sql("SELECT * FROM vehicles;", conn)
    users = pd.read_sql("SELECT * FROM users;", conn)

    conn.close()

    carac.columns = carac.columns.str.lower()
    places.columns = places.columns.str.lower()
    vehicles.columns = vehicles.columns.str.lower()
    users.columns = users.columns.str.lower()

    df = users.merge(vehicles, on=["num_acc","num_veh"], how="left")\
              .merge(carac, on="num_acc", how="left")\
              .merge(places, on="num_acc", how="left")

    df = df.drop_duplicates()
    df = df.dropna()
    df = df[~df.isin(["-"]).any(axis=1)]

    X = df.drop(columns=["num_acc","num_veh","adr","lat","long", "dep","place","voie","v1","v2","pr", "lartpc", "larrout","occutc","obs","grav"])

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)
    X = X.astype(int)

    model = joblib.load("xgb_model.pkl")

    preds = model.predict(X)

    df["prediction"] = preds

    print("\n===== SAMPLE PREDICTIONS =====")
    print(df[["num_acc", "num_veh", "prediction"]].head(10))

    return {
    "status": "prediction completed",
    "n_predictions": len(preds),
    "sample_predictions": df[["num_acc", "num_veh", "prediction"]]
        .head(10)
        .to_dict(orient="records")
}


if __name__ == "__main__":
    result = predict_model()
    print(result)