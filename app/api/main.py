from fastapi import FastAPI
from app.api.routes import correction, model

app = FastAPI(title="Multilingual Correction LLM API", version="0.1.0")

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

app.include_router(correction.router)
app.include_router(model.router)
