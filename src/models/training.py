def train_model():
    import pandas as pd
    import psycopg2
    import xgboost as xgb
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    from sklearn.utils.class_weight import compute_sample_weight
    import mlflow
    import mlflow.xgboost
    from mlflow.tracking import MlflowClient

    MLFLOW_TRACKING_URI = "http://localhost:8080"
    MODEL_NAME          = "gravite-accident"
    METRIC_KEY          = "f1_macro"

    conn_params = {
        "dbname": "mlops_accidents",
        "user": "postgres",
        "password": "admin",
        "host": "localhost",
        "port": 5432
    }

    n_estimators     = 400
    max_depth        = 6
    learning_rate    = 0.01
    subsample        = 0.8
    colsample_bytree = 0.8
    eval_metric      = "logloss"
    tree_method      = "hist"
    n_jobs           = -1
    random_state     = 42
    verbosity        = 0
    
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
        n_estimators     = n_estimators,
        max_depth        = max_depth,
        learning_rate    = learning_rate,
        subsample        = subsample,
        colsample_bytree = colsample_bytree,
        eval_metric      = eval_metric,
        tree_method      = tree_method,
        n_jobs           = n_jobs,
        random_state     = random_state,
        verbosity        = verbosity
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

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("accidentologie-gravite")
    client = MlflowClient()  

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Params
        mlflow.log_param("n_estimators",     n_estimators)
        mlflow.log_param("max_depth",        max_depth)
        mlflow.log_param("learning_rate",    learning_rate)
        mlflow.log_param("subsample",        subsample)
        mlflow.log_param("colsample_bytree", colsample_bytree)
        mlflow.log_param("eval_metric",      eval_metric)
        mlflow.log_param("tree_method",      tree_method)
        mlflow.log_param("n_train",          len(X_train))
        mlflow.log_param("n_test",           len(X_test))

        # Métriques
        mlflow.log_metric("accuracy", round(float(acc), 4))
        mlflow.log_metric("f1_macro", round(float(f1),  4))

        # Modèle
        mlflow.xgboost.log_model(model, artifact_path="model")

        # Sauvegarde locale conservée
        joblib.dump(model, "xgb_model.pkl")

        # Comparaison & promotion
        mv          = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        new_version = mv.version

        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])

        if not prod_versions:
            client.transition_model_version_stage(MODEL_NAME, new_version, "Production")
            print(f"\n[Registry] v{new_version} → Production (premier modèle, F1={f1:.4f})")
        else:
            prod_f1 = client.get_run(prod_versions[0].run_id).data.metrics.get(METRIC_KEY, 0.0)
            delta   = f1 - prod_f1
            mlflow.log_metric("delta_f1", round(delta, 4))

            if f1 > prod_f1:
                client.transition_model_version_stage(MODEL_NAME, prod_versions[0].version, "Archived")
                client.transition_model_version_stage(MODEL_NAME, new_version, "Production")
                print(f"\n[Registry] v{new_version} → Production (+{delta:.4f}) | v{prod_versions[0].version} → Archived")
            else:
                client.transition_model_version_stage(MODEL_NAME, new_version, "Staging")
                print(f"\n[Registry] v{new_version} → Staging (inférieur de {abs(delta):.4f})")

    print(f"\n[MLflow] UI → http://localhost:8080")

    return {"status": "training completed",
    "accuracy": float(acc),
    "f1_macro": float(f1)
    }

if __name__ == "__main__":
    result = train_model()
    print(result)