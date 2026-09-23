"""
FastAPI application entry point for the Multilingual Correction LLM API.

Lifespan management:
    - On startup: initializes the CorrectionEngine singleton and loads the
      model (or enters mock mode if CORRECTION_MOCK_MODE is set).
    - On shutdown: no special cleanup required (GC handles model release).
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.dependencies import (
    get_correction_engine,
    initialize_engine,
    load_engine_model,
)
from app.api.routes import correction, model

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize and load the correction engine."""
    try:
        engine = initialize_engine()
        load_engine_model()
        logger.info(
            "CorrectionEngine ready — model=%s mock=%s",
            engine.model_id_or_path,
            engine.mock_mode,
        )
    except RuntimeError as e:
        # Log the error but allow the app to start so /health can report it
        logger.error("Failed to load correction model: %s", e)
    except FileNotFoundError as e:
        logger.error("Missing configuration: %s", e)

    yield  # application runs here


app = FastAPI(
    title="Multilingual Correction LLM API",
    version="0.2.0",
    description=(
        "REST API for multilingual OCR text correction. "
        "Supports English, Hindi, and code-mixed text."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict:
    """
    Returns application health status including model readiness.

    Response fields:
        - status: 'ok' if engine is ready, 'degraded' if not
        - model_loaded: whether the model weights are in memory
        - mock_mode: whether the engine is running in mock mode
        - model_id: identifier of the configured model
    """
    try:
        engine = get_correction_engine()
        return {
            "status": "ok",
            "model_loaded": engine.model is not None or engine.mock_mode,
            "mock_mode": engine.mock_mode,
            "model_id": engine.model_id_or_path,
        }
    except RuntimeError:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "model_loaded": False,
                "mock_mode": False,
                "model_id": None,
                "detail": "CorrectionEngine not available.",
            },
        )


app.include_router(correction.router)
app.include_router(model.router)
