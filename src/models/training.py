def train_model():
    import pandas as pd
    import psycopg2
    import xgboost as xgb
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_sample_weight

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

    df = users.merge(vehicles, on=["num_acc","num_veh"], how="left")\
              .merge(carac, on="num_acc", how="left")\
              .merge(places, on="num_acc", how="left")

    df = df.drop_duplicates().dropna()
    df = df[~df.isin(["-"]).any(axis=1)]

    y = df["grav"].astype(int)
    y = y - 1
    X = df.drop(columns=["num_acc","num_veh","adr","lat","long", "dep","place","voie","v1","v2","pr", "lartpc", "larrout","occutc","obs","grav"])

    # Encodage catégories
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category").cat.codes
        
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)
    X = X.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBClassifier(
        n_estimators     = 400,
        max_depth        = 8,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        eval_metric      = "logloss",
        tree_method      = "hist",
        n_jobs           = -1,
        random_state     = 42,
        verbosity        = 0
    )

    sample_weights = compute_sample_weight(
                    class_weight="balanced",
                    y=y_train
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred, average="macro")

    print("\n===== METRICS =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score (macro) : {f1:.4f}")

    print("\n===== CONFUSION MATRIX =====")
    print(confusion_matrix(y_test, y_pred))

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "xgb_model.pkl")

    return {"status": "training completed",
    "accuracy": float(acc),
    "f1_macro": float(f1)
    }

if __name__ == "__main__":
    result = train_model()
    print(result)