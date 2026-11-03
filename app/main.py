import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.inference import predecir
from app.model_loader import cargar_modelo
from app.schemas import PredictionRequest, PredictionResponse

ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ml["modelo"], ml["version"] = cargar_modelo()
    except Exception as error:
        print("No se pudo cargar el modelo:", error)
        ml["modelo"] = None
        ml["version"] = None
    yield
    ml.clear()


app = FastAPI(title="Churn Prediction API", lifespan=lifespan)


@app.get("/health")
def health():
    if ml.get("modelo") is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    return {"status": "ok", "model_version": ml["version"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if ml.get("modelo") is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    resultado = predecir(ml["modelo"], request.model_dump())
    resultado["model_version"] = ml["version"]
    resultado["request_id"] = str(uuid.uuid4())
    return resultado
