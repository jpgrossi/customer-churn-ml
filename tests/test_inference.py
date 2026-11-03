import pytest

from app.inference import calcular_nivel_riesgo, predecir


def test_modelo_carga_y_tiene_predict_proba(modelo):
    assert hasattr(modelo, "predict_proba")


def test_modelo_conoce_las_clases_yes_y_no(modelo):
    assert set(modelo.classes_) == {"Yes", "No"}


def test_prediccion_valida_tiene_el_contrato_esperado(modelo, cliente_valido):
    resultado = predecir(modelo, cliente_valido)

    assert set(resultado.keys()) == {"prediction", "probability", "risk_level"}
    assert resultado["prediction"] in ["Yes", "No"]
    assert isinstance(resultado["probability"], float)
    assert 0 <= resultado["probability"] <= 1
    assert resultado["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_prediccion_con_total_charges_vacio(modelo, cliente_valido):
    cliente_valido["TotalCharges"] = None
    resultado = predecir(modelo, cliente_valido)

    assert 0 <= resultado["probability"] <= 1


@pytest.mark.parametrize("probabilidad, nivel_esperado", [
    (0.10, "LOW"),
    (0.39, "LOW"),
    (0.40, "MEDIUM"),
    (0.69, "MEDIUM"),
    (0.70, "HIGH"),
    (0.95, "HIGH"),
])
def test_nivel_de_riesgo_segun_umbrales(probabilidad, nivel_esperado):
    assert calcular_nivel_riesgo(probabilidad) == nivel_esperado
