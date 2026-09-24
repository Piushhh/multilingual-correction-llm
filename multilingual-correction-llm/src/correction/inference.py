"""
Correction inference engine.

The CorrectionEngine orchestrates: prompt building → model generation →
change detection → structured output. It delegates model operations to
a ModelAdapter, making it agnostic to the underlying LLM.

Backward compatibility:
    CorrectionEngine(config, mock_mode=True)   # still works
    CorrectionEngine(config, adapter=my_adapter)  # new adapter-based API
"""

from typing import Any, Dict, List, Optional

from src.correction.change_detection import detect_changes
from src.correction.model_adapter import (
    GemmaBaselineAdapter,
    MockAdapter,
    ModelAdapter,
)
from src.correction.prompts import PromptFormatter


def _detect_device() -> str:
    """Detect available device without requiring torch at import time."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class CorrectionEngine:
    """
    Inference engine for multilingual text correction.

    Supports pluggable model backends via the ModelAdapter interface.
    The engine itself handles prompt formatting, change detection, and
    output structuring — the adapter handles only raw text generation.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        mock_mode: bool = False,
        adapter: Optional[ModelAdapter] = None,
    ):
        """
        Initialize the CorrectionEngine.

        Args:
            config: Model and generation configuration dict.
            mock_mode: If True, uses MockAdapter (no real model loaded).
            adapter: Optional explicit ModelAdapter. If provided, mock_mode
                     is ignored and this adapter is used directly.
        """
        self.config = config
        self.mock_mode = mock_mode

        # Adapter resolution: explicit > mock_mode > default (Gemma baseline)
        if adapter is not None:
            self._adapter = adapter
            self.mock_mode = isinstance(adapter, MockAdapter)
        elif mock_mode:
            self._adapter = MockAdapter()
        else:
            self._adapter = GemmaBaselineAdapter(config)

        self.model_id_or_path = self._adapter.model_id
        self.device = config.get("device", _detect_device())
        self.max_length = config.get("max_length", 512)

        self.formatter = PromptFormatter()

        # Backward-compat attributes used by existing routes and tests
        self.model = None
        self.tokenizer = None
        self.generator = None

    def load_model(self) -> None:
        """Load the model via the adapter."""
        if self.mock_mode:
            print("Running in explicit mock mode. Model loading skipped.")
            return

        self._adapter.load()

        # Set self.model to a truthy sentinel so existing code that checks
        # `engine.model is not None` continues to work.
        if self._adapter.is_loaded():
            self.model = self._adapter

    def build_prompt(self, text: str, language: str, domain: str) -> str:
        """Build the correction prompt."""
        return self.formatter.build_prompt(text, language, domain)

    def generate(self, prompt: str) -> str:
        """Generate corrected text from the model."""
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
        language = request.get("language", "en")
        domain = request.get("domain", "general")

        prompt = self.build_prompt(text, language, domain)
        corrected_text = self.generate(prompt)

        if self.mock_mode and "[MOCK GENERATED CORRECTION]" in corrected_text:
            # Deterministic mock: apply simple known corrections for testing
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
