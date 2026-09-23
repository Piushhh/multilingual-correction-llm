"""
Tests for app/api/dependencies.py

Covers:
    - Singleton/cached engine behavior
    - Explicit mock mode via environment variable
    - Model-loading failure propagation
    - Missing or invalid configuration handling
    - Thread-safety of initialization
"""

import os
import threading
import pytest

from app.api.dependencies import (
    get_correction_engine,
    initialize_engine,
    load_engine_model,
    reset_engine,
    _resolve_config,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset engine singleton and environment before each test."""
    reset_engine()
    # Save and restore environment
    saved = {
        k: os.environ.get(k)
        for k in ("CORRECTION_MOCK_MODE", "CORRECTION_MODEL_CONFIG")
    }
    yield
    reset_engine()
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ── Singleton behavior ───────────────────────────────────────────────

class TestSingleton:
    """The engine must be initialized exactly once."""

    def test_initialize_returns_same_instance(self):
        """Calling initialize_engine() twice returns the same object."""
        os.environ["CORRECTION_MOCK_MODE"] = "true"
        engine1 = initialize_engine()
        engine2 = initialize_engine()
        assert engine1 is engine2

    def test_get_before_initialize_raises(self):
        """get_correction_engine() raises if no engine was initialized."""
        with pytest.raises(RuntimeError, match="not available"):
            get_correction_engine()

    def test_get_after_initialize_returns_engine(self):
        """get_correction_engine() returns the initialized engine."""
        os.environ["CORRECTION_MOCK_MODE"] = "true"
        expected = initialize_engine()
        actual = get_correction_engine()
        assert actual is expected

    def test_reset_clears_singleton(self):
        """reset_engine() allows re-initialization."""
        os.environ["CORRECTION_MOCK_MODE"] = "true"
        engine1 = initialize_engine()
        reset_engine()
        engine2 = initialize_engine()
        assert engine1 is not engine2

    def test_thread_safety(self):
        """Concurrent initialize_engine() calls produce the same instance."""
        os.environ["CORRECTION_MOCK_MODE"] = "true"
        engines = []
        errors = []

        def init():
            try:
                engines.append(initialize_engine())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=init) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(set(id(e) for e in engines)) == 1


# ── Mock mode ────────────────────────────────────────────────────────

class TestMockMode:
    """Mock mode must be explicitly enabled via environment."""

    def test_mock_mode_true(self):
        """Engine enters mock mode when CORRECTION_MOCK_MODE=true."""
        os.environ["CORRECTION_MOCK_MODE"] = "true"
        engine = initialize_engine()
        assert engine.mock_mode is True

    def test_mock_mode_false_by_default(self):
        """Engine defaults to production mode (mock_mode=False)."""
        os.environ.pop("CORRECTION_MOCK_MODE", None)
        engine = initialize_engine()
        assert engine.mock_mode is False

    def test_mock_mode_yes_variant(self):
        """CORRECTION_MOCK_MODE=yes is accepted."""
        os.environ["CORRECTION_MOCK_MODE"] = "yes"
        engine = initialize_engine()
        assert engine.mock_mode is True

    def test_mock_mode_1_variant(self):
        """CORRECTION_MOCK_MODE=1 is accepted."""
        os.environ["CORRECTION_MOCK_MODE"] = "1"
        engine = initialize_engine()
        assert engine.mock_mode is True

    def test_mock_mode_load_skips_model(self):
        """In mock mode, load_engine_model() is a no-op."""
        os.environ["CORRECTION_MOCK_MODE"] = "true"
        initialize_engine()
        load_engine_model()  # Should not raise
        engine = get_correction_engine()
        assert engine.model is None  # No real model loaded


# ── Model-loading failure ────────────────────────────────────────────

class TestModelLoadingFailure:
    """Model-loading errors must propagate clearly."""

    def test_load_without_initialize_raises(self):
        """load_engine_model() raises if engine was never initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            load_engine_model()

    def test_production_load_failure_propagates(self):
        """In production mode, loading a nonexistent model raises RuntimeError."""
        os.environ.pop("CORRECTION_MOCK_MODE", None)
        engine = initialize_engine()
        # The default config points to google/gemma-2b-it which is not locally
        # available. This should raise a RuntimeError.
        with pytest.raises(RuntimeError):
            load_engine_model()

    def test_load_failure_recorded_in_get(self):
        """After a load failure, get_correction_engine() also raises."""
        os.environ.pop("CORRECTION_MOCK_MODE", None)
        initialize_engine()
        try:
            load_engine_model()
        except RuntimeError:
            pass  # Expected

        with pytest.raises(RuntimeError, match="failed to initialize"):
            get_correction_engine()


# ── Configuration ────────────────────────────────────────────────────

class TestConfiguration:
    """Configuration loading from file and environment."""

    def test_default_config_loads(self):
        """Default config/training_config.yaml loads successfully."""
        config = _resolve_config()
        assert "model_name" in config
        assert config["model_name"] == "google/gemma-2b-it"

    def test_custom_config_path(self, tmp_path):
        """CORRECTION_MODEL_CONFIG overrides the default path."""
        custom = tmp_path / "custom.yaml"
        custom.write_text('model_name: "custom-test-model"\nmax_length: 256\n')
        os.environ["CORRECTION_MODEL_CONFIG"] = str(custom)
        config = _resolve_config()
        assert config["model_name"] == "custom-test-model"
        assert config["max_length"] == 256

    def test_missing_config_raises(self, tmp_path):
        """FileNotFoundError raised when config file does not exist."""
        os.environ["CORRECTION_MODEL_CONFIG"] = str(tmp_path / "nonexistent.yaml")
        os.environ.pop("CORRECTION_MOCK_MODE", None)
        with pytest.raises(FileNotFoundError, match="not found"):
            _resolve_config()

    def test_missing_config_in_mock_mode_ok(self):
        """In mock mode, missing config is acceptable."""
        os.environ["CORRECTION_MODEL_CONFIG"] = "/nonexistent/path.yaml"
        os.environ["CORRECTION_MOCK_MODE"] = "true"
        engine = initialize_engine()
        assert engine.mock_mode is True
        assert engine.model_id_or_path == "mock-baseline"

    def test_missing_config_in_production_mode_raises(self):
        """In production mode, missing config is a fatal error."""
        os.environ["CORRECTION_MODEL_CONFIG"] = "/nonexistent/path.yaml"
        os.environ.pop("CORRECTION_MOCK_MODE", None)
        with pytest.raises(FileNotFoundError):
            initialize_engine()
