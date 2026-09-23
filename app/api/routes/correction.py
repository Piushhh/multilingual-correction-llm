"""
Correction API routes.

POST /correct       — Correct a single text
POST /correct/batch — Correct multiple texts (max 50)

Mock mode is a property of the engine singleton (set at startup via
CORRECTION_MOCK_MODE). Individual requests cannot toggle mock mode at
runtime — this prevents production traffic from silently receiving
fabricated corrections.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_correction_engine
from app.schemas.correction import (
    BatchCorrectionRequest,
    BatchCorrectionResponse,
    CorrectionRequest,
    CorrectionResponse,
)
from src.correction.inference import CorrectionEngine

router = APIRouter(prefix="/correct", tags=["correction"])


def _assert_engine_ready(engine: CorrectionEngine) -> None:
    """Raises HTTP 503 if the engine cannot serve real traffic."""
    if not engine.mock_mode and engine.model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. The server started but the correction "
                "model is not available. Check /health for details."
            ),
        )


@router.post("", response_model=CorrectionResponse)
def correct_text(
    request: CorrectionRequest,
    engine: CorrectionEngine = Depends(get_correction_engine),
):
    """
    Correct a single text input.

    In production mode the real model is used. In mock mode (server started
    with CORRECTION_MOCK_MODE=true) a deterministic mock correction is
    returned. The per-request `mock_mode` field is ignored for safety —
    mock behavior is controlled only at the server level.
    """
    _assert_engine_ready(engine)

    try:
        result = engine.correct(request.model_dump(exclude={"mock_mode"}))
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchCorrectionResponse)
def correct_batch(
    request: BatchCorrectionRequest,
    engine: CorrectionEngine = Depends(get_correction_engine),
):
    """
    Correct a batch of text inputs (maximum 50).

    Uses the same shared engine instance — the model is NOT reloaded
    for each item. Mock mode is controlled at the server level.
    """
    if len(request.requests) > 50:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds limit of 50.",
        )

    _assert_engine_ready(engine)

    responses = []
    for req in request.requests:
        try:
            result = engine.correct(req.model_dump(exclude={"mock_mode"}))
            responses.append(result)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {"responses": responses}
