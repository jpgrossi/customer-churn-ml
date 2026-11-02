import subprocess

import pandas as pd
import yaml
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.config import ID_COL, TARGET, RANDOM_STATE, TEST_SIZE
from src.features.build_pipeline import build_preprocessor

DATA_PATH = "data/raw/customer_churn_historical.csv"
DVC_FILE = "data/raw/customer_churn_historical.csv.dvc"
EXPERIMENT_NAME = "churn-classification"


def obtener_git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def obtener_hash_dataset():
    with open(DVC_FILE) as archivo:
        contenido = yaml.safe_load(archivo)
    return contenido["outs"][0]["md5"]


def entrenar_y_registrar(nombre, modelo, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=nombre):
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", modelo)
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        indice_yes = list(pipeline.classes_).index("Yes")
        y_proba = pipeline.predict_proba(X_test)[:, indice_yes]

        metricas = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label="Yes", zero_division=0),
            "recall": recall_score(y_test, y_pred, pos_label="Yes"),
            "f1": f1_score(y_test, y_pred, pos_label="Yes"),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        matriz = confusion_matrix(y_test, y_pred, labels=["No", "Yes"])

        mlflow.log_param("modelo", type(modelo).__name__)
        mlflow.log_params(modelo.get_params())
        mlflow.log_param("split_random_state", RANDOM_STATE)
        mlflow.log_param("test_size", TEST_SIZE)

        mlflow.set_tag("git_sha", obtener_git_sha())
        mlflow.set_tag("dataset", DATA_PATH)
        mlflow.set_tag("dataset_md5", obtener_hash_dataset())

        mlflow.log_metrics(metricas)
        mlflow.log_dict({"labels": ["No", "Yes"], "matriz": matriz.tolist()}, "confusion_matrix.json")
        mlflow.sklearn.log_model(pipeline, name="model", input_example=X_train.head(3), skops_trusted_types=["numpy.dtype"])

        print(nombre, "- recall:", round(metricas["recall"], 3), "- roc_auc:", round(metricas["roc_auc"], 3))


def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET, ID_COL])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    entrenar_y_registrar("baseline", DummyClassifier(strategy="most_frequent"), X_train, X_test, y_train, y_test)
    entrenar_y_registrar("logreg", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), X_train, X_test, y_train, y_test)
    entrenar_y_registrar("logreg_balanced", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE), X_train, X_test, y_train, y_test)
    entrenar_y_registrar("rf", RandomForestClassifier(random_state=RANDOM_STATE), X_train, X_test, y_train, y_test)
    entrenar_y_registrar("rf_depth8", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE), X_train, X_test, y_train, y_test)
    entrenar_y_registrar("rf_balanced", RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE), X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()

