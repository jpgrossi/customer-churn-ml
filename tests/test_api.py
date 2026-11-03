import pytest


def test_health_responde_ok(client):
    respuesta = client.get("/health")

    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "ok"
    assert isinstance(respuesta.json()["model_version"], str)


def test_predict_con_cliente_valido(client, cliente_valido):
    respuesta = client.post("/predict", json=cliente_valido)
    cuerpo = respuesta.json()

    assert respuesta.status_code == 200
    assert set(cuerpo.keys()) == {"prediction", "probability", "risk_level", "model_version", "request_id"}
    assert cuerpo["prediction"] in ["Yes", "No"]
    assert 0 <= cuerpo["probability"] <= 1
    assert cuerpo["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(cuerpo["model_version"], str)
    assert len(cuerpo["request_id"]) > 0


def test_cada_pedido_tiene_un_request_id_distinto(client, cliente_valido):
    primera = client.post("/predict", json=cliente_valido).json()
    segunda = client.post("/predict", json=cliente_valido).json()

    assert primera["request_id"] != segunda["request_id"]


@pytest.mark.parametrize("campo, valor_invalido", [
    ("tenure", -1),
    ("Contract", "Weekly"),
    ("MonthlyCharges", "alto"),
    ("SeniorCitizen", 2),
    ("InternetService", None),
])
def test_predict_rechaza_valores_invalidos(client, cliente_valido, campo, valor_invalido):
    cliente_valido[campo] = valor_invalido
    respuesta = client.post("/predict", json=cliente_valido)

    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["loc"][-1] == campo


def test_predict_rechaza_campo_faltante(client, cliente_valido):
    del cliente_valido["gender"]
    respuesta = client.post("/predict", json=cliente_valido)

    assert respuesta.status_code == 422
