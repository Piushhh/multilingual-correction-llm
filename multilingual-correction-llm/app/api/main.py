from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.core_llm.inference.generate import LLM

app = FastAPI(title="Multilingual Correction LLM API", version="0.1.0")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "base" / "best.pt"
DEFAULT_TOKENIZER = ROOT / "data" / "tokenizer" / "tokenizer.json"


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.8
    top_k: int | None = 40


class GenerateResponse(BaseModel):
    text: str


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    """
    Loads the trained checkpoint + tokenizer once per process.
    This is Member 1's model wired up behind this project's own API --
    no external API key needed, since it's serving your own trained model.
    """
    if not DEFAULT_CHECKPOINT.exists() or not DEFAULT_TOKENIZER.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model found. Run the tokenizer training and "
                "pretrain.py scripts first (see src/core_llm/README.md), "
                "then retry."
            ),
        )
    return LLM.from_checkpoint(
        checkpoint_path=str(DEFAULT_CHECKPOINT),
        tokenizer_path=str(DEFAULT_TOKENIZER),
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Generate text from the trained model. This is a minimal starting point
    for Member 3's correction pipeline/API integration -- swap in a
    correction-specific prompt format here once that pipeline exists.
    """
    llm = get_llm()
    text = llm.generate(
        request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
    )
    return GenerateResponse(text=text)
