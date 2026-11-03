import pandas as pd
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.config import ID_COL, TARGET, RANDOM_STATE, TEST_SIZE, RISK_THRESHOLD_MEDIUM, RISK_THRESHOLD_HIGH

UMBRAL_MEDIUM = RISK_THRESHOLD_MEDIUM
UMBRAL_HIGH = RISK_THRESHOLD_HIGH


def main():
    load_dotenv()

    df = pd.read_csv("data/raw/customer_churn_historical.csv")
    X = df.drop(columns=[TARGET, ID_COL])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    modelo = mlflow.sklearn.load_model("models:/CustomerChurnModel@candidate")
    indice_yes = list(modelo.classes_).index("Yes")
    proba = modelo.predict_proba(X_test)[:, indice_yes]

    resultado = pd.DataFrame({"proba": proba, "churn_real": (y_test == "Yes").values})
    resultado["nivel"] = pd.cut(
        resultado["proba"],
        bins=[0, UMBRAL_MEDIUM, UMBRAL_HIGH, 1.0001],
        labels=["LOW", "MEDIUM", "HIGH"],
        right=False
    )

    resumen = resultado.groupby("nivel", observed=True).agg(
        clientes=("proba", "count"),
        tasa_churn=("churn_real", "mean")
    )
    resumen["porcentaje_clientes"] = resumen["clientes"] / len(resultado)
    print(resumen.round(3))


if __name__ == "__main__":
    main()

