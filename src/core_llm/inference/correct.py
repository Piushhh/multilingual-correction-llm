"""
Correction inference interface.

Provides CorrectionInference wrapping LLMInference for the specific
task of text correction via the prompt template:

    Correct:
    <incorrect text>
    Answer:

Usage:

    from src.core_llm.inference.correct import CorrectionInference

    corrector = CorrectionInference.from_checkpoint("path/to/correction_model.pt")

    result = corrector.correct(
        text="Deep learnig is a subfeld of machne learning.",
        max_new_tokens=64,
    )
    # result is the corrected text string

NOTE: Correction quality depends entirely on training. A freshly initialized
or lightly trained model will NOT produce meaningful corrections.
Do not claim correction quality without empirical evaluation.
"""

from pathlib import Path
from typing import Optional

from src.core_llm.inference.generate import LLMInference


# Default paths (relative to repo root)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_CORRECTION_CKPT = _REPO_ROOT / "src" / "core_llm" / "checkpoints" / "correction_model.pt"
_DEFAULT_TOKENIZER = _REPO_ROOT / "src" / "core_llm" / "tokenizer" / "tokenizer.model"

PROMPT_PREFIX = "Correct:\n"
ANSWER_PREFIX = "\nAnswer:\n"


class CorrectionInference:
    """
    Correction inference wrapper.

    Formats the correction prompt, generates a response via LLMInference,
    and returns only the generated correction text (not the prompt).
    """

    def __init__(self, llm: LLMInference):
        """
        Args:
            llm: An initialized LLMInference instance.
        """
        self._llm = llm

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "CorrectionInference":
        """
        Load a CorrectionInference from a checkpoint.

        Args:
            checkpoint_path: Path to correction_model.pt (or domain_adapted_model.pt).
                             Defaults to src/core_llm/checkpoints/correction_model.pt
            tokenizer_path:  Optional override for tokenizer path.
            device:          'cpu', 'cuda', or None (auto-detect).

        Returns:
            Initialized CorrectionInference instance.

        Raises:
            FileNotFoundError: If checkpoint or tokenizer is missing.
            ValueError:        If vocab_size mismatch.
        """
        if checkpoint_path is None:
            checkpoint_path = str(_DEFAULT_CORRECTION_CKPT)

        llm = LLMInference.from_checkpoint(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
            device=device,
        )

        return cls(llm)

    # ── Correction ────────────────────────────────────────────────────────

    def correct(
        self,
        text: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Correct the given text.

        Args:
            text:           The text to be corrected.
            max_new_tokens: Maximum tokens to generate as the correction.
            temperature:    Sampling temperature (0.0 = greedy by default).
            top_k:          Top-k sampling. None = no filtering.

        Returns:
            The correction as a string (prompt is excluded from output).
            Returns empty string if input is empty.

        NOTE: The quality of correction depends on training.
        A freshly initialized or lightly trained model will not produce
        meaningful corrections. This interface is structurally correct
        regardless of model quality.
        """
        if not text:
            return ""

        # Format correction prompt
        prompt = PROMPT_PREFIX + text + ANSWER_PREFIX

        correction = self._llm.generate(
            text=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_at_eos=True,
        )

        return correction.strip()

    @property
    def llm(self) -> LLMInference:
        """Access the underlying LLMInference instance."""
        return self._llm


# ── Script entry point ────────────────────────────────────────────────────────

def main():
    import sys

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "Deep learnig is a subfeld of machne learning."

    print("Loading correction model...")

    try:
        corrector = CorrectionInference.from_checkpoint()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nBLOCKED: No correction checkpoint available.")
        print("Run training pipeline first.")
        raise SystemExit(1)

    print(f"Input: {text}")
    result = corrector.correct(text, max_new_tokens=64, temperature=0.0)
    print(f"Corrected: {result}")


if __name__ == "__main__":
    main()
