"""
/generate router — Member 1 LLM direct-generation endpoint.

Exposes the trained custom LLM for text generation without the full
correction pipeline. Useful for development, evaluation, and debugging.

Unlike /correct (which uses CorrectionEngine + Gemma baseline or mock),
/generate uses CustomLLMAdapter (Member 1's transformer, Member 1 code).

The model is loaded lazily on first request via get_custom_llm().
Returns 503 if checkpoint or tokenizer is missing.
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(prefix="/generate", tags=["generate"])

_ROOT = Path(__file__).resolve().parents[3]  # repo root
_DEFAULT_CHECKPOINT = _ROOT / "checkpoints" / "domain" / "best.pt"
_DEFAULT_TOKENIZER = _ROOT / "data" / "tokenizer" / "tokenizer.json"

# Module-level singleton — loaded once per process on first /generate request
_custom_llm_adapter = None


def _get_custom_llm_adapter():
    """
    Return the loaded CustomLLMAdapter singleton.

    Raises:
        HTTPException(503): if checkpoint or tokenizer file is missing,
            or if the checkpoint lacks the 'config' key.
    """
    global _custom_llm_adapter
    if _custom_llm_adapter is not None:
        return _custom_llm_adapter

    from src.integration.adapters.custom_llm_adapter import CustomLLMAdapter

    adapter = CustomLLMAdapter(
        checkpoint_path=str(_DEFAULT_CHECKPOINT),
        tokenizer_path=str(_DEFAULT_TOKENIZER),
    )

    try:
        adapter.load()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Custom LLM checkpoint or tokenizer not found: {exc}. "
                "Run src/core_llm training scripts first."
            ),
        ) from exc
    except (KeyError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load custom LLM: {exc}",
        ) from exc

    _custom_llm_adapter = adapter
    return adapter


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    max_new_tokens: int = Field(default=80, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(default=40, ge=1)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    text: str
    model_id: str
    prompt_length: int
    generated_length: int


@router.post("", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Generate text continuation with the Member 1 custom LLM.

    Returns the continuation only (prompt is stripped from the output).
    Note: this model is a from-scratch prototype trained on limited data
    and is NOT instruction-tuned — quality should NOT be compared to
    production LLMs.
    """
    adapter = _get_custom_llm_adapter()

    continuation = adapter.generate(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
    )

    return GenerateResponse(
        text=continuation,
        model_id=adapter.model_id,
        prompt_length=len(request.prompt),
        generated_length=len(continuation),
    )
