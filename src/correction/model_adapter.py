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
    Concrete adapter for Member 1's custom-trained PyTorch Transformer model.
    Connects to src/core_llm/checkpoints/correction_model.pt and SentencePiece BPETokenizer.

    IMPORTANT CONTEXT LENGTH HANDLING:
    Member 1's custom model was trained on a 64-token context length with prompt format:
    "Correct:\n{text}\nAnswer:\n"
    This adapter extracts the core text from verbose prompts (such as Member 3's PromptFormatter)
    to guarantee that prompt + generation fit strictly within the 64-token context window.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._model_path = self.config.get(
            "custom_model_path",
            "src/core_llm/checkpoints/correction_model.pt",
        )
        self._tokenizer_path = self.config.get(
            "tokenizer_path",
            "src/core_llm/tokenizer/tokenizer.model",
        )
        self._device_str = self.config.get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self._model = None
        self._tokenizer = None
        self._device = None

    def load(self) -> None:
        from src.core_llm.inference.correct import load_model

        device = torch.device(self._device_str)
        self._model, self._tokenizer, self._device = load_model(
            checkpoint_path=self._model_path,
            tokenizer_path=self._tokenizer_path,
            device=device,
        )

    def _extract_input_text(self, prompt: str) -> str:
        """
        Extract the core sentence to correct from complex prompt formats to fit 64 tokens.
        """
        if "INPUT:\n" in prompt:
            parts = prompt.split("INPUT:\n", 1)[1]
            if "\n\nOUTPUT:" in parts:
                return parts.split("\n\nOUTPUT:", 1)[0].strip()
            elif "\nOUTPUT:" in parts:
                return parts.split("\nOUTPUT:", 1)[0].strip()
            return parts.strip()
        elif prompt.startswith("Correct:\n") and "\nAnswer:\n" in prompt:
            return prompt.split("Correct:\n", 1)[1].split("\nAnswer:\n", 1)[0].strip()
        return prompt.strip()

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_loaded():
            raise RuntimeError("Custom model is not loaded. Call load() first.")

        from src.core_llm.inference.correct import correct_text

        text_to_correct = self._extract_input_text(prompt)
        max_new_tokens = kwargs.get("max_new_tokens", 30)

        return correct_text(
            self._model,
            self._tokenizer,
            self._device,
            text_to_correct,
            max_new_tokens=max_new_tokens,
        )

    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def model_id(self) -> str:
        return f"member1-custom-llm ({self._model_path})"

    @property
    def is_baseline(self) -> bool:
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
