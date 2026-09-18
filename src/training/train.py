import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from src.config import ID_COL, TARGET, RANDOM_STATE, TEST_SIZE, MODEL_DIR, MODEL_FILE
from src.features.build_pipeline import build_preprocessor


def main():
    df = pd.read_csv("data/raw/customer_churn_historical.csv")

    X = df.drop(columns=[TARGET, ID_COL])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])

    pipeline.fit(X_train, y_train)

   
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("Precision:", round(precision_score(y_test, y_pred, pos_label="Yes"), 3))
    print("Recall:", round(recall_score(y_test, y_pred, pos_label="Yes"), 3))
    print("F1:", round(f1_score(y_test, y_pred, pos_label="Yes"), 3))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 3))

    ruta_modelo = MODEL_DIR + "/" + MODEL_FILE
    joblib.dump(pipeline, ruta_modelo)
    print("Modelo guardado en:", ruta_modelo)


if __name__ == "__main__":
    main()
