"""
Model information API routes.

GET /model/info — Returns current model configuration and status.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_correction_engine
from src.correction.inference import CorrectionEngine

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info")
def get_model_info(
    engine: CorrectionEngine = Depends(get_correction_engine),
):
    """
    Returns metadata about the currently loaded correction model.

    The `baseline` field indicates whether the configured model is the
    project's initial baseline (google/gemma-2b-it) rather than a
    production-trained model. This will be updated when Member 1's
    custom model architecture is integrated.
    """
    is_baseline = "gemma" in engine.model_id_or_path.lower()

    return {
        "model_id": engine.model_id_or_path,
        "device": engine.device,
        "loaded": engine.model is not None or engine.mock_mode,
        "mock_mode": engine.mock_mode,
        "baseline": is_baseline,
        "note": (
            "This is the baseline model (google/gemma-2b-it). "
            "It will be replaced when Member 1's custom model is ready."
            if is_baseline
            else "Production model loaded."
        ),
    }
