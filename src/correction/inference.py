from typing import Dict, Any, Optional
import torch

# Assuming transformers is available, otherwise we gracefully degrade
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from src.correction.prompts import PromptFormatter
from src.correction.change_detection import detect_changes

class CorrectionEngine:
    """
    Baseline inference engine for multilingual correction.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Correction Engine with a configuration.
        """
        self.config = config
        self.model_id_or_path = config.get("model_name", "google/gemma-2b-it") # default fallback
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = config.get("max_length", 512)
        
        self.formatter = PromptFormatter()
        self.model = None
        self.tokenizer = None
        self.generator = None

    def load_model(self):
        """Loads the model and tokenizer into memory."""
        if not TRANSFORMERS_AVAILABLE:
            print("Warning: transformers library not found. Running in mock mode.")
            return

        print(f"Loading model {self.model_id_or_path} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id_or_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.generator = pipeline(
                "text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to mock mode.")
            self.model = None

    def build_prompt(self, text: str, language: str, domain: str) -> str:
        """Builds the prompt for the model."""
        return self.formatter.build_prompt(text, language, domain)

    def generate(self, prompt: str) -> str:
        """Generates raw text from the model."""
        if not self.model or not self.generator:
            # Mock behavior if model failed to load or transformers is missing
            return f"{prompt}\n[MOCK GENERATED CORRECTION]"
            
        outputs = self.generator(
            prompt, 
            max_new_tokens=self.max_length,
            do_sample=self.config.get("do_sample", False),
            temperature=self.config.get("temperature", 0.0),
            return_full_text=False
        )
        return outputs[0]['generated_text'].strip()

    def detect_changes(self, original: str, corrected: str) -> list:
        """Detects structured changes between original and corrected text."""
        return detect_changes(original, corrected)

    def correct(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to process a correction request.
        Expected request: {"text": "...", "language": "en", "domain": "general"}
        """
        text = request.get("text", "")
        language = request.get("language", "en")
        domain = request.get("domain", "general")
        
        prompt = self.build_prompt(text, language, domain)
        
        # We assume the generated text is the pure corrected text 
        # (this requires proper prompt tuning in production)
        corrected_text = self.generate(prompt)
        
        # If in mock mode, just simulate a correction for testing
        if "[MOCK GENERATED CORRECTION]" in corrected_text:
            corrected_text = text.replace("teh", "the") if "teh" in text else text
            
        changes = self.detect_changes(text, corrected_text)
        
        return {
            "corrected_text": corrected_text,
            "changes": changes,
            "metadata": {
                "language": language,
                "domain": domain
            }
        }
