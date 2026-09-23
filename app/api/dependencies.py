import yaml
from src.correction.inference import CorrectionEngine

_engine = None

def get_correction_engine():
    global _engine
    if _engine is None:
        try:
            with open("config/training_config.yaml", "r") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            config = {"model_name": "mock"}
            
        # Initialize engine but wait for explicit load calls
        _engine = CorrectionEngine(config, mock_mode=False)
        # Note: In production, load_model() would be called during app startup
    return _engine
