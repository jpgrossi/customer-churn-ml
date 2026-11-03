from fastapi import FastAPI

app = FastAPI(title="Churn Prediction API")


@app.get("/health")
def health():
    return {"status": "ok"}
