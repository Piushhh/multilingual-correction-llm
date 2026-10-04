"""
Model adapter interface for the Multilingual Correction System.

Defines the abstract contract that any LLM backend must satisfy to be
used by the CorrectionEngine. This decouples the correction pipeline
from any specific model (Gemma, Member 1's custom architecture, etc.).

Concrete adapters:
    - GemmaBaselineAdapter:  HuggingFace Gemma (google/gemma-2b-it)
    - MockAdapter:           Deterministic mock for testing
    - CustomLLMAdapter:      Placeholder for Member 1's model (not yet implemented)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ModelAdapter(ABC):
    """
    Abstract interface that all model backends must implement.

    Member 1's custom model, Gemma baseline, or any future model should
    subclass this adapter. The CorrectionEngine delegates all model
    operations to the adapter.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Load model weights and tokenizer into memory.

        Raises:
            RuntimeError: If loading fails (e.g., missing weights, OOM).
        """

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate corrected text from a prompt.

        Args:
            prompt: The formatted correction prompt.
            **kwargs: Model-specific generation parameters.

        Returns:
            The generated text (correction only, not the prompt).

        Raises:
            RuntimeError: If the model is not loaded.
        """

    @abstractmethod
    def is_loaded(self) -> bool:
        """Returns True if the model is ready for inference."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Human-readable identifier for this model backend."""

    @property
    @abstractmethod
    def is_baseline(self) -> bool:
        """True if this is a baseline/placeholder, not a production model."""


class MockAdapter(ModelAdapter):
    """
    Deterministic mock adapter for testing.

    Returns predictable outputs so tests can validate pipeline logic
    without loading real model weights.
    """

    def load(self) -> None:
        """No-op: mock adapter has no weights to load."""
        pass

    def generate(self, prompt: str, **kwargs) -> str:
        """Returns a sentinel string that CorrectionEngine recognizes."""
        return "[MOCK GENERATED CORRECTION]"

    def is_loaded(self) -> bool:
        return True

    @property
    def model_id(self) -> str:
        return "mock-adapter"

    @property
    def is_baseline(self) -> bool:
        return True


class GemmaBaselineAdapter(ModelAdapter):
    """
    Baseline adapter using google/gemma-2b-it via HuggingFace.

    This is the initial baseline model for the project. It will be
    replaced when Member 1 delivers a custom-trained model.

    Config keys used:
        - model_name:  HuggingFace model ID (default: google/gemma-2b-it)
        - device:      'cuda' or 'cpu'
        - max_length:  Maximum generation length
        - do_sample:   Whether to use sampling
        - temperature: Sampling temperature
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
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError:
            raise RuntimeError(
                "GemmaBaselineAdapter requires 'torch' and 'transformers'. "
                "Install them or use MockAdapter for testing."
            )

        try:
            import torch as _torch

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=_torch.float16 if self._device == "cuda" else _torch.float32,
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


class CustomLLMAdapter(ModelAdapter):
    """
    Adapter interface for Member 1's custom-trained model.

    ┌─────────────────────────────────────────────────────────────┐
    │  NOT YET IMPLEMENTED — WAITING ON MEMBER 1                  │
    │                                                             │
    │  When Member 1 delivers their model, implement this class:  │
    │  1. Set model_path to their checkpoint directory            │
    │  2. Implement load() using their model class                │
    │  3. Implement generate() using their inference interface    │
    │  4. Set is_baseline to False                                │
    └─────────────────────────────────────────────────────────────┘

    Expected Member 1 deliverables:
        - A model class in src/core_llm/model/
        - A tokenizer in src/core_llm/tokenizer/
        - A checkpoint in checkpoints/ or a HuggingFace model ID
        - An inference interface in src/core_llm/inference/

    This adapter will wrap their interface to satisfy the ModelAdapter
    contract used by CorrectionEngine.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._model_path = config.get("custom_model_path", None)

    def load(self) -> None:
        raise NotImplementedError(
            "CustomLLMAdapter is not yet implemented. "
            "Member 1's model architecture (src/core_llm/) has not been delivered. "
            "Use GemmaBaselineAdapter or MockAdapter until Member 1's code is available."
        )

    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError(
            "CustomLLMAdapter.generate() requires Member 1's inference interface."
        )

    def is_loaded(self) -> bool:
        return False

    @property
    def model_id(self) -> str:
        return self._model_path or "custom-llm (not configured)"

    @property
    def is_baseline(self) -> bool:
        return False
