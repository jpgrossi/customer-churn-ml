import mlflow.sklearn
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.config import MODEL_NAME, MODEL_ALIAS


def cargar_modelo():
    load_dotenv()
    client = MlflowClient()

    version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
    modelo = mlflow.sklearn.load_model("models:/" + MODEL_NAME + "/" + str(version))

    return modelo, str(version)
