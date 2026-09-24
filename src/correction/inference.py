"""
Correction inference engine.

The CorrectionEngine orchestrates: prompt building -> model generation ->
change detection -> structured output. It delegates model operations to
a ModelAdapter, making it agnostic to the underlying LLM.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

from src.correction.change_detection import detect_changes
from src.correction.model_adapter import (
    CustomLLMAdapter,
    GemmaBaselineAdapter,
    MockAdapter,
    ModelAdapter,
)
from src.correction.prompts import PromptFormatter


def _detect_device() -> str:
    """Detect available device without requiring external dependencies."""
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class CorrectionEngine:
    """
    Inference engine for multilingual text correction.
    Supports pluggable model backends via the ModelAdapter interface.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        mock_mode: bool = False,
        adapter: Optional[ModelAdapter] = None,
    ):
        """
        Initialize the CorrectionEngine.

        Args:
            config: Model and generation configuration dict.
            mock_mode: If True, uses MockAdapter (no real weights loaded).
            adapter: Optional explicit ModelAdapter. If provided, mock_mode
                     is derived from the adapter type.
        """
        self.config = config or {}
        self.mock_mode = mock_mode

        # Adapter resolution priority: explicit adapter > mock_mode > custom_llm > baseline
        if adapter is not None:
            self._adapter = adapter
            self.mock_mode = isinstance(adapter, MockAdapter)
        elif mock_mode:
            self._adapter = MockAdapter()
        elif (
            self.config.get("model_name") == "custom"
            or self.config.get("model_type") == "custom"
            or self.config.get("use_custom", False)
            or "custom_model_path" in self.config
        ):
            # Custom LLM explicitly requested (Member 1's custom PyTorch model)
            self._adapter = CustomLLMAdapter(self.config)
        else:
            # Baseline adapter (Gemma baseline or empty config placeholder)
            self._adapter = GemmaBaselineAdapter(self.config)

        self.model_id_or_path = self._adapter.model_id
        self.device = self.config.get("device", _detect_device())
        self.max_length = self.config.get("max_length", 64)

        self.formatter = PromptFormatter()

        # Backward-compat attributes
        self.model = None
        self.tokenizer = None
        self.generator = None

    def load_model(self) -> None:
        """Load the model via the configured adapter."""
        if self.mock_mode:
            return

        self._adapter.load()

        if self._adapter.is_loaded():
            self.model = self._adapter

    def build_prompt(self, text: str, language: str, domain: str) -> str:
        """Build the correction prompt."""
        return self.formatter.build_prompt(text, language, domain)

    def generate(self, prompt: str) -> str:
        """Generate corrected text from the model adapter."""
        if self.mock_mode:
            return self._adapter.generate(prompt)

        if not self._adapter.is_loaded():
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        return self._adapter.generate(prompt)

    def detect_changes(self, original: str, corrected: str) -> List[Dict[str, Any]]:
        """Detect structured changes between original and corrected text."""
        return detect_changes(original, corrected)

    def correct(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a correction request.

        Args:
            request: Dict with keys 'text', 'language' (default 'en'),
                     'domain' (default 'general').

        Returns:
            Dict with 'corrected_text', 'changes', and 'metadata'.
        """
        text = request.get("text", "")
        if not text:
            return {
                "corrected_text": "",
                "changes": [],
                "metadata": {
                    "language": request.get("language", "en"),
                    "domain": request.get("domain", "general"),
                },
            }

        language = request.get("language", "en")
        domain = request.get("domain", "general")

        prompt = self.build_prompt(text, language, domain)
        corrected_text = self.generate(prompt)

        if self.mock_mode and "[MOCK GENERATED CORRECTION]" in corrected_text:
            # Deterministic mock rule for test consistency
            corrected_text = text.replace("teh", "the") if "teh" in text else text

        changes = self.detect_changes(text, corrected_text)

        return {
            "corrected_text": corrected_text,
            "changes": changes,
            "metadata": {
                "language": language,
                "domain": domain,
            },
        }

    @property
    def adapter(self) -> ModelAdapter:
        """Access the underlying model adapter."""
        return self._adapter

    @property
    def is_baseline(self) -> bool:
        """True if the current adapter is a baseline/placeholder."""
        return self._adapter.is_baseline
