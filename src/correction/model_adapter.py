"""
Model adapter interface for the Multilingual Correction System.

Defines the abstract contract that any LLM backend must satisfy to be
used by the CorrectionEngine. Decouples the correction pipeline from specific models.

Concrete adapters:
    - MockAdapter: Deterministic mock for fast unit testing without loading weights.
    - CustomLLMAdapter: Adapter bridging to Member 1's custom PyTorch Transformer.
    - GemmaBaselineAdapter: Hugging Face Gemma baseline (google/gemma-2b-it).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import torch


class ModelAdapter(ABC):
    """
    Abstract interface that all model backends must implement.
    The CorrectionEngine delegates all model operations to this adapter.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights and tokenizer into memory."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate corrected text from a prompt."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Returns True if the model is loaded and ready for inference."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Human-readable identifier for this model backend."""

    @property
    @abstractmethod
    def is_baseline(self) -> bool:
        """True if this is a baseline/placeholder, False for production custom model."""


class MockAdapter(ModelAdapter):
    """
    Deterministic mock adapter for testing.
    Returns predictable outputs so tests can validate pipeline logic
    without requiring heavy model weights.
    """

    def load(self) -> None:
        pass

    def generate(self, prompt: str, **kwargs) -> str:
        return "[MOCK GENERATED CORRECTION]"

    def is_loaded(self) -> bool:
        return True

    @property
    def model_id(self) -> str:
        return "mock-adapter"

    @property
    def is_baseline(self) -> bool:
        return True


class CustomLLMAdapter(ModelAdapter):
    """
    Adapter for Member 1's custom-trained CausalTransformerLM.

    This adapter connects the CorrectionEngine to Member 1's inference
    interface without duplicating any Transformer code.

    Integration chain:
        CorrectionEngine
            → CustomLLMAdapter
                → LLMInference.from_checkpoint()
                    → CausalTransformerLM
                        → BPETokenizer + checkpoint

    Config keys:
        custom_model_path (str):   Path to correction checkpoint (.pt file)
        tokenizer_path    (str):   Optional tokenizer path override
        device            (str):   'cpu' or 'cuda' (default: auto-detect)
        max_new_tokens    (int):   Maximum new tokens to generate (default: 64)
        temperature       (float): Sampling temperature (default: 0.0 = greedy)
        top_k             (int):   Top-k sampling filter (default: None)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._model_path = self.config.get("custom_model_path", None)
        self._tokenizer_path = self.config.get(
            "tokenizer_path",
            "src/core_llm/tokenizer/tokenizer.model",
        )
        self._device_str = self.config.get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self._max_new_tokens = self.config.get("max_new_tokens", 64)
        self._temperature = self.config.get("temperature", 0.0)
        self._top_k = self.config.get("top_k", None)
        self._llm = None  # set after load()

    def load(self) -> None:
        """
        Load Member 1's model from checkpoint using LLMInference.

        Raises:
            RuntimeError:       If custom_model_path is not configured.
            FileNotFoundError:  If checkpoint or tokenizer file is missing.
            ValueError:         If vocab_size mismatch detected.
            KeyError:           If checkpoint is missing model_config.
        """
        if self._model_path is None:
            raise RuntimeError(
                "CustomLLMAdapter: 'custom_model_path' is not set in config.\n"
                "Set config['custom_model_path'] to the path of the correction checkpoint.\n"
                "Example: 'src/core_llm/checkpoints/correction_model.pt'"
            )

        try:
            from src.core_llm.inference.generate import LLMInference
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import Member 1's inference interface: {e}\n"
                "Ensure src/core_llm/ is properly installed."
            )

        try:
            self._llm = LLMInference.from_checkpoint(
                checkpoint_path=self._model_path,
                tokenizer_path=self._tokenizer_path,
                device=self._device_str,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"CustomLLMAdapter: checkpoint or tokenizer not found.\n{e}"
            )
        except (KeyError, ValueError) as e:
            raise RuntimeError(
                f"CustomLLMAdapter: failed to load Member 1 model.\n{e}"
            )

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Member 1's LLMInference.

        Args:
            prompt: The formatted correction prompt string.
            **kwargs: Optional overrides for max_new_tokens, temperature, top_k.

        Returns:
            Generated text string (prompt excluded).

        Raises:
            RuntimeError: If load() has not been called successfully.
        """
        if self._llm is None:
            raise RuntimeError(
                "CustomLLMAdapter: model is not loaded. Call load() first."
            )

        max_new_tokens = kwargs.get("max_new_tokens", self._max_new_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        top_k = kwargs.get("top_k", self._top_k)

        return self._llm.generate(
            text=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_at_eos=True,
        )

    def is_loaded(self) -> bool:
        """Returns True if the model has been successfully loaded."""
        return self._llm is not None

    @property
    def model_id(self) -> str:
        return f"member1-custom-llm ({self._model_path})"

    @property
    def is_baseline(self) -> bool:
        """Always False — this is Member 1's custom model, not a baseline."""
        return False


class GemmaBaselineAdapter(ModelAdapter):
    """
    Baseline adapter using google/gemma-2b-it via Hugging Face.
    Used as an external reference baseline.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._model_name = config.get("model_name", "google/gemma-2b-it")
        self._device = config.get("device", "cpu")
        self._max_length = config.get("max_length", 512)
        self._model = None
        self._tokenizer = None
        self._generator = None

    def load(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError:
            raise RuntimeError(
                "GemmaBaselineAdapter requires 'transformers'. "
                "Install it or use CustomLLMAdapter/MockAdapter."
            )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            ).to(self._device)
            self._generator = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device=0 if self._device == "cuda" else -1,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Gemma baseline: {e}")

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        outputs = self._generator(
            prompt,
            max_new_tokens=self._max_length,
            do_sample=self.config.get("do_sample", False),
            temperature=self.config.get("temperature", 0.0),
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip()

    def is_loaded(self) -> bool:
        return self._model is not None and self._generator is not None

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def is_baseline(self) -> bool:
        return True
