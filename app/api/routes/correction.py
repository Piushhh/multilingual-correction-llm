from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.schemas.correction import CorrectionRequest, CorrectionResponse, BatchCorrectionRequest, BatchCorrectionResponse
from app.api.dependencies import get_correction_engine
from src.correction.inference import CorrectionEngine

router = APIRouter(prefix="/correct", tags=["correction"])

@router.post("", response_model=CorrectionResponse)
def correct_text(request: CorrectionRequest, engine: CorrectionEngine = Depends(get_correction_engine)):
    # Temporarily set mock mode based on request (safe baseline)
    original_mock = engine.mock_mode
    engine.mock_mode = request.mock_mode
    
    try:
        if not engine.model and not engine.mock_mode:
            raise HTTPException(status_code=503, detail="Model not loaded. Cannot serve real traffic.")
            
        result = engine.correct(request.model_dump())
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        engine.mock_mode = original_mock

@router.post("/batch", response_model=BatchCorrectionResponse)
def correct_batch(request: BatchCorrectionRequest, engine: CorrectionEngine = Depends(get_correction_engine)):
    responses = []
    
    # We enforce batch size limits in production
    if len(request.requests) > 50:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit of 50.")
        
    for req in request.requests:
        original_mock = engine.mock_mode
        engine.mock_mode = req.mock_mode
        try:
            if not engine.model and not engine.mock_mode:
                raise HTTPException(status_code=503, detail="Model not loaded.")
            result = engine.correct(req.model_dump())
            responses.append(result)
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            engine.mock_mode = original_mock
            
    return {"responses": responses}
