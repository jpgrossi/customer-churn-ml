import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_loader import cargar_modelo
from app.schemas import PredictionRequest


@pytest.fixture(scope="session")
def modelo():
    modelo_cargado, version = cargar_modelo()
    return modelo_cargado


@pytest.fixture
def cliente_valido():
    return dict(PredictionRequest.model_config["json_schema_extra"]["example"])


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client
