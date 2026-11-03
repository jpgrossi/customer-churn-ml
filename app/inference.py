import pandas as pd

from src.config import RISK_THRESHOLD_MEDIUM, RISK_THRESHOLD_HIGH


def calcular_nivel_riesgo(probabilidad):
    if probabilidad >= RISK_THRESHOLD_HIGH:
        return "HIGH"
    if probabilidad >= RISK_THRESHOLD_MEDIUM:
        return "MEDIUM"
    return "LOW"


def predecir(modelo, datos_cliente):
    X = pd.DataFrame([datos_cliente])
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"])

    indice_yes = list(modelo.classes_).index("Yes")
    probabilidad = float(modelo.predict_proba(X)[0][indice_yes])
    prediccion = str(modelo.predict(X)[0])

    return {
        "prediction": prediccion,
        "probability": probabilidad,
        "risk_level": calcular_nivel_riesgo(probabilidad)
    }
