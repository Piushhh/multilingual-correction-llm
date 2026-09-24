"""
Application-level dependency for the CorrectionEngine singleton.

The engine is initialized once at application startup via `initialize_engine()`
and reused across all requests via `get_correction_engine()`.

Configuration priority:
    1. Environment variable CORRECTION_MODEL_CONFIG (path to YAML config)
    2. Default path: config/training_config.yaml
    3. Environment variable CORRECTION_MOCK_MODE=true to force mock mode

The model is NOT loaded automatically. Call `load_engine_model()` during
application startup (e.g., in a FastAPI lifespan event) to load weights.
"""

import os
import threading
from pathlib import Path
from typing import Optional

import yaml

from src.correction.inference import CorrectionEngine


# Module-level singleton, protected by a lock for thread safety
_engine: Optional[CorrectionEngine] = None
_engine_lock = threading.Lock()
_initialization_error: Optional[str] = None


def _resolve_config() -> dict:
    """
    Loads the correction model configuration from a YAML file.

    Resolution order:
        1. CORRECTION_MODEL_CONFIG environment variable (absolute path)
        2. config/training_config.yaml (project default)

    Raises FileNotFoundError if neither source is available and mock mode
    is not explicitly enabled.
    """
    config_path = os.environ.get(
        "CORRECTION_MODEL_CONFIG",
        "config/training_config.yaml",
    )

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model configuration file not found: {path.resolve()}. "
            "Set CORRECTION_MODEL_CONFIG to a valid YAML path, or create "
            "config/training_config.yaml."
        )

    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}

    return config


def _is_mock_mode() -> bool:
    """Returns True if mock mode is explicitly enabled via environment."""
    return os.environ.get("CORRECTION_MOCK_MODE", "").lower() in ("true", "1", "yes")


def initialize_engine() -> CorrectionEngine:
    """
    Creates the singleton CorrectionEngine instance.

    Thread-safe. If the engine has already been initialized, returns the
    existing instance. If initialization fails, the error is recorded and
    subsequent calls to `get_correction_engine()` will raise it.

    Returns:
        The initialized CorrectionEngine instance.

    Raises:
        RuntimeError: If initialization fails (e.g., missing config in
            non-mock mode).
    """
    global _engine, _initialization_error

    with _engine_lock:
        if _engine is not None:
            return _engine

        mock_mode = _is_mock_mode()

        try:
            config = _resolve_config()
        except FileNotFoundError:
            if mock_mode:
                # In explicit mock mode, a missing config is acceptable
                config = {"model_name": "mock-baseline"}
            else:
                raise

        _engine = CorrectionEngine(config, mock_mode=mock_mode)
        _initialization_error = None
        return _engine


def load_engine_model() -> None:
    """
    Loads the model weights into the singleton engine.

    Should be called once during application startup (e.g., in a FastAPI
    lifespan event). In mock mode, this is a no-op.

    Raises:
        RuntimeError: If the engine has not been initialized, or if model
            loading fails.
    """
    global _initialization_error

    if _engine is None:
        raise RuntimeError(
            "CorrectionEngine not initialized. Call initialize_engine() first."
        )

    try:
        _engine.load_model()
    except RuntimeError as e:
        _initialization_error = str(e)
        raise


def get_correction_engine() -> CorrectionEngine:
    """
    FastAPI dependency that provides the shared CorrectionEngine instance.

    This NEVER creates a new engine. The engine must be initialized during
    application startup. If initialization failed, this raises a clear error.

    Returns:
        The shared CorrectionEngine singleton.

    Raises:
        RuntimeError: If the engine was never initialized or if initialization
            previously failed.
    """
    if _engine is None:
        raise RuntimeError(
            "CorrectionEngine not available. The application did not "
            "initialize the engine at startup."
        )

    if _initialization_error is not None:
        raise RuntimeError(
            f"CorrectionEngine failed to initialize: {_initialization_error}"
        )

    return _engine


def reset_engine() -> None:
    """
    Resets the singleton engine. For testing only.

    This allows tests to create fresh engine instances without leaking state
    between test cases.
    """
    global _engine, _initialization_error
    with _engine_lock:
        _engine = None
        _initialization_error = None
