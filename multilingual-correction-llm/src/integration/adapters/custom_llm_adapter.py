"""
CustomLLMAdapter — wraps Member 1's from-scratch Transformer LLM through
the Member 3 ModelAdapter interface.

IMPORTANT LIMITATIONS (must be communicated clearly):
  - This is a SMALL from-scratch prototype (~44MB weights) trained on a
    LIMITED corpus (data/raw/train.txt + data/domain/).
  - It is NOT instruction-tuned and will NOT reliably follow correction
    instructions. It continues text in a language-model fashion.
  - It should NOT be compared to Gemma, GPT, or any instruction-tuned model.
  - The adapter only EXPOSES the trained LLM through the existing interface.
    Meaningful correction quality requires Member 3 to fine-tune the model
    on correction data (starter_dataset.jsonl) — that is NOT done yet.
  - is_baseline = False (this is Member 1's custom architecture, not Gemma),
    but "not baseline" does NOT mean "production quality."
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import — torch is heavy; don't pay the cost unless this adapter is used
_LLM = None


def _get_llm_class():
    global _LLM
    if _LLM is None:
        from src.core_llm.inference.generate import LLM
        _LLM = LLM
    return _LLM


class CustomLLMAdapter:
    """
    ModelAdapter subclass wrapping src.core_llm.inference.generate.LLM.

    Conforms to the same interface as GemmaBaselineAdapter and MockAdapter
    in src/correction/model_adapter.py so the CorrectionEngine can use it
    interchangeably.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ):
        """
        Args:
            checkpoint_path: path to a .pt checkpoint saved by save_checkpoint().
                Falls back to CUSTOM_CHECKPOINT env var, then
                checkpoints/domain/best.pt.
            tokenizer_path: path to a tokenizer.json file.
                Falls back to CUSTOM_TOKENIZER env var, then
                data/tokenizer/tokenizer.json.
        """
        self._checkpoint_path = (
            checkpoint_path
            or os.environ.get("CUSTOM_CHECKPOINT")
            or "checkpoints/domain/best.pt"
        )
        self._tokenizer_path = (
            tokenizer_path
            or os.environ.get("CUSTOM_TOKENIZER")
            or "data/tokenizer/tokenizer.json"
        )
        self._llm = None
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    # ModelAdapter interface                                               #
    # ------------------------------------------------------------------ #

    @property
    def model_id(self) -> str:
        """Stable identifier derived from the checkpoint path."""
        return f"custom-llm:{Path(self._checkpoint_path).name}"

    @property
    def is_baseline(self) -> bool:
        """False — this is Member 1's custom architecture, not Gemma."""
        return False

    def load(self) -> None:
        """
        Load the checkpoint and tokenizer.

        Raises:
            FileNotFoundError: if checkpoint or tokenizer file is missing.
            KeyError: if the checkpoint dict is missing the 'config' key
                (checkpoint was not saved with save_checkpoint()).
            RuntimeError: on any other load failure.
        """
        ckpt_path = Path(self._checkpoint_path)
        tok_path = Path(self._tokenizer_path)

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"CustomLLMAdapter: checkpoint not found at {ckpt_path.resolve()}. "
                "Run pretraining first: "
                "python -m src.core_llm.training.pretrain "
                "--train-file data/raw/train.txt --val-file data/raw/val.txt "
                "--tokenizer data/tokenizer/tokenizer.json "
                "--checkpoint-dir checkpoints/base"
            )

        if not tok_path.exists():
            raise FileNotFoundError(
                f"CustomLLMAdapter: tokenizer not found at {tok_path.resolve()}. "
                "Run: python -m src.core_llm.tokenizer.train_tokenizer"
            )

        # Validate the checkpoint contains 'config' before loading the model
        import torch
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        if "config" not in ckpt:
            raise KeyError(
                f"CustomLLMAdapter: checkpoint at {ckpt_path} is missing the "
                "'config' key. This checkpoint was not saved with save_checkpoint() "
                "and cannot be used to rebuild the model automatically. "
                "Retrain with the current pretrain.py."
            )

        try:
            LLM = _get_llm_class()
            self._llm = LLM.from_checkpoint(
                checkpoint_path=str(ckpt_path),
                tokenizer_path=str(tok_path),
            )
            logger.info(
                "CustomLLMAdapter loaded: checkpoint=%s config=%s",
                ckpt_path,
                ckpt["config"],
            )
        except Exception as exc:
            self._load_error = str(exc)
            raise RuntimeError(
                f"CustomLLMAdapter: failed to load checkpoint {ckpt_path}: {exc}"
            ) from exc

    def is_loaded(self) -> bool:
        return self._llm is not None

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text continuation for `prompt`.

        Returns ONLY the continuation (not the prompt), so it fits cleanly
        into the CorrectionEngine workflow that compares output to input.

        Honors:
            max_new_tokens (int, default 80)
            temperature    (float, default 1.0; 0 = greedy)
            top_k          (int or None)
            top_p          (float or None)

        Left-truncation:
            If prompt + max_new_tokens exceeds max_seq_len (256), the prompt
            is left-truncated to (max_seq_len - max_new_tokens) tokens,
            preserving the most recent text. When truncation occurs it is
            logged as a warning. Note: left-truncation may drop the
            TASK/RULES header before the INPUT text.
        """
        if not self.is_loaded():
            raise RuntimeError(
                "CustomLLMAdapter.generate() called before load(). "
                "Call load() first or use the engine's load_model() in startup."
            )

        max_new_tokens: int = int(kwargs.get("max_new_tokens", 80))
        temperature: float = float(kwargs.get("temperature", 1.0))
        top_k = kwargs.get("top_k", None)
        top_p = kwargs.get("top_p", None)

        # --- Left-truncation ---
        max_seq_len: int = self._llm.model.config.max_seq_len
        max_new_tokens = min(max_new_tokens, max_seq_len - 1)

        prompt_ids = self._llm.tokenizer.encode(prompt).ids
        max_prompt_tokens = max_seq_len - max_new_tokens

        if len(prompt_ids) > max_prompt_tokens:
            logger.warning(
                "CustomLLMAdapter: prompt (%d tokens) exceeds budget (%d). "
                "Left-truncating — TASK/RULES header may be dropped.",
                len(prompt_ids),
                max_prompt_tokens,
            )
            prompt_ids = prompt_ids[-max_prompt_tokens:]
            # Decode the truncated prompt back to text for LLM.generate()
            prompt = self._llm.tokenizer.decode(prompt_ids)

        # LLM.generate returns full decoded text (prompt + continuation).
        # We strip the prompt prefix to return continuation only.
        full_text: str = self._llm.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k is not None else None,
            top_p=top_p if top_p is not None else None,
        )

        # Strip prompt from the beginning of the generated text.
        # BPE decoding often prepends a leading space or normalizes whitespace,
        # so check against raw prompt, decoded prompt, and trimmed variants.
        decoded_prompt = self._llm.tokenizer.decode(prompt_ids)
        if full_text.startswith(prompt):
            continuation = full_text[len(prompt):]
        elif full_text.startswith(decoded_prompt):
            continuation = full_text[len(decoded_prompt):]
        elif full_text.lstrip().startswith(prompt.lstrip()):
            offset = full_text.find(prompt.lstrip())
            continuation = full_text[offset + len(prompt.lstrip()):]
        elif full_text.lstrip().startswith(decoded_prompt.lstrip()):
            offset = full_text.find(decoded_prompt.lstrip())
            continuation = full_text[offset + len(decoded_prompt.lstrip()):]
        else:
            # Tokenize boundary mismatch — return full text as fallback
            continuation = full_text

        return continuation.strip()

