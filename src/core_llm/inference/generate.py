"""
LLMInference — clean public inference interface for Member 1's model.

Usage from any module:

    from src.core_llm.inference.generate import LLMInference

    llm = LLMInference.from_checkpoint("path/to/checkpoint.pt")

    result = llm.generate(
        text="Deep learning is a subfield",
        max_new_tokens=64,
        temperature=1.0,
        top_k=50,
    )
    # result is a str containing only the newly generated text

    # Or get full output including the prompt
    full = llm.generate_full("Deep learning", max_new_tokens=32)

This module never relies on CLI arguments. It is designed to be imported
and called programmatically by other modules (model_adapter, tests, API).
"""

from pathlib import Path
from typing import Optional

import torch

from src.core_llm.model.config import ModelConfig
from src.core_llm.model.transformer import CausalTransformerLM
from src.core_llm.tokenizer.bpe_tokenizer import BPETokenizer


class LLMInference:
    """
    Clean inference wrapper for CausalTransformerLM.

    Responsibilities:
        - Load model and tokenizer from a checkpoint
        - Provide a generate() method that returns only new tokens
        - Handle context truncation, EOS stopping, temperature, top-k
        - Work on CPU and CUDA without caller changes

    Never modify this class to import CLI arguments.
    """

    def __init__(
        self,
        model: CausalTransformerLM,
        tokenizer: BPETokenizer,
        device: torch.device,
        model_config: ModelConfig,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._config = model_config

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "LLMInference":
        """
        Load a CausalTransformerLM from a checkpoint file.

        Args:
            checkpoint_path: Path to a .pt checkpoint file produced by
                             pretrain.py, domain_adapt.py, or correction_finetune.py
            tokenizer_path:  Optional override for tokenizer path. If None,
                             the path stored in the checkpoint is used.
            device:          'cpu', 'cuda', or None (auto-detect).

        Returns:
            Initialized LLMInference instance.

        Raises:
            FileNotFoundError: If checkpoint or tokenizer file is missing.
            KeyError:          If checkpoint is missing required keys.
            ValueError:        If vocab_size mismatch between model and tokenizer.
        """
        checkpoint_path = str(checkpoint_path)

        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Run training first or verify the checkpoint path."
            )

        if device is None:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            _device = torch.device(device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=_device,
            weights_only=False,
        )

        # ── Resolve model config ──────────────────────────────────────────
        if "model_config" not in checkpoint:
            raise KeyError(
                f"Checkpoint '{checkpoint_path}' is missing 'model_config'. "
                "This checkpoint was produced by an older training script. "
                "Re-run training to produce a properly formatted checkpoint."
            )

        config_data = checkpoint["model_config"]
        if isinstance(config_data, dict):
            config = ModelConfig.from_dict(config_data)
        elif isinstance(config_data, ModelConfig):
            config = config_data
        else:
            raise TypeError(
                f"Unexpected type for model_config in checkpoint: {type(config_data)}"
            )

        # ── Resolve tokenizer path ────────────────────────────────────────
        if tokenizer_path is None:
            tokenizer_path = checkpoint.get("tokenizer_path")

        if tokenizer_path is None:
            # Fall back to default location relative to this file
            default = Path(__file__).resolve().parents[2] / "tokenizer" / "tokenizer.model"
            tokenizer_path = str(default)

        if not Path(str(tokenizer_path)).is_file():
            raise FileNotFoundError(
                f"Tokenizer not found: {tokenizer_path}\n"
                "Run: python -m src.core_llm.scripts.train_tokenizer"
            )

        tokenizer = BPETokenizer(str(tokenizer_path))

        # ── Validate vocab consistency ────────────────────────────────────
        if config.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f"Model vocab_size ({config.vocab_size}) != "
                f"tokenizer vocab_size ({tokenizer.vocab_size}).\n"
                "The checkpoint and tokenizer were produced by different training runs."
            )

        # ── Build model ───────────────────────────────────────────────────
        model = CausalTransformerLM(config).to(_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return cls(model, tokenizer, _device, config)

    # ── Inference ─────────────────────────────────────────────────────────

    def generate(
        self,
        text: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,  # reserved, not yet implemented
        stop_at_eos: bool = True,
    ) -> str:
        """
        Generate new text from a prompt.

        Only the newly generated portion is returned (not the prompt).

        Args:
            text:           Prompt text (UTF-8, English/Hindi/mixed).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature:    Sampling temperature. 0.0 = greedy.
            top_k:          Top-k sampling filter. None = no filtering.
            top_p:          Reserved for nucleus sampling (not yet implemented).
            stop_at_eos:    Stop generation when EOS token is produced.

        Returns:
            Newly generated text as a string (prompt excluded).

        Raises:
            ValueError: If max_new_tokens < 1 or temperature < 0.
        """
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if not text:
            return ""

        prompt_ids = self._tokenizer.encode(text)

        if not prompt_ids:
            return ""

        # Truncate prompt to leave room for generated tokens
        max_prompt_len = self._config.context_length - 1
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=self._device
        )

        original_length = input_ids.shape[1]

        eos_id = self._tokenizer.EOS_ID if stop_at_eos else None

        output_ids = self._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_id,
        )

        # Return only newly generated tokens
        new_ids = output_ids[0, original_length:].tolist()
        return self._tokenizer.decode(new_ids)

    def generate_full(
        self,
        text: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Generate text and return the full output (prompt + generated).

        Args:
            text:           Prompt text.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature:    Sampling temperature. 0.0 = greedy.
            top_k:          Top-k sampling. None = no filtering.

        Returns:
            Full decoded string including prompt.
        """
        if not text:
            return ""

        prompt_ids = self._tokenizer.encode(text)
        max_prompt_len = self._config.context_length - 1
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=self._device
        )

        output_ids = self._model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=self._tokenizer.EOS_ID,
        )

        return self._tokenizer.decode(output_ids[0].tolist())

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self._config.vocab_size

    @property
    def context_length(self) -> int:
        return self._config.context_length

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model_config(self) -> ModelConfig:
        return self._config


# ── Backward-compat script functions ─────────────────────────────────────────
# (for code that calls generate_text(model, tokenizer, device, prompt) directly)

def load_model(checkpoint_path: str = None, tokenizer_path: str = None):
    """
    Legacy helper: returns (model, tokenizer, device).
    Prefer LLMInference.from_checkpoint() in new code.
    """
    from src.core_llm.model.config import ModelConfig
    from src.core_llm.model.transformer import CausalTransformerLM
    from src.core_llm.tokenizer.bpe_tokenizer import BPETokenizer

    REPO_ROOT = Path(__file__).resolve().parents[4]

    if tokenizer_path is None:
        tokenizer_path = str(REPO_ROOT / "src" / "core_llm" / "tokenizer" / "tokenizer.model")

    if checkpoint_path is None:
        checkpoint_path = str(
            REPO_ROOT / "src" / "core_llm" / "checkpoints" / "domain_adapted_model.pt"
        )

    llm = LLMInference.from_checkpoint(checkpoint_path, tokenizer_path)
    return llm._model, llm._tokenizer, llm._device


def generate_text(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int = 30,
) -> str:
    """
    Legacy helper for backward compatibility.
    Returns full text (prompt + generated).
    """
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
    )

    return tokenizer.decode(output_ids[0].tolist())
