from fastapi import APIRouter, Depends, HTTPException
from app.api.dependencies import get_correction_engine
from src.correction.inference import CorrectionEngine

router = APIRouter(prefix="/model", tags=["model"])

@router.get("/info")
def get_model_info(engine: CorrectionEngine = Depends(get_correction_engine)):
    return {
        "model_id": engine.model_id_or_path,
        "device": engine.device,
        "loaded": engine.model is not None,
        "mock_mode_default": engine.mock_mode
    }
