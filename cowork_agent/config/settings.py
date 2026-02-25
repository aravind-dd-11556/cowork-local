"""
Configuration loader — YAML file + environment variable overrides.
"""

from __future__ import annotations
import os
import yaml
from pathlib import Path
from typing import Any, Optional


class Config:
    """Configuration container with dot-access and env var support."""

    def __init__(self, data: dict):
        self._data = data

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by dot-separated key path."""
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a config value by dot-separated key path."""
        keys = key.split(".")
        d = self._data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    @property
    def raw(self) -> dict:
        return self._data

    def __repr__(self) -> str:
        return f"Config({self._data})"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file with env var overrides.

    Priority (highest to lowest):
    1. Environment variables (COWORK_LLM_PROVIDER, etc.)
    2. User config file (if provided)
    3. Default config

    Env var mapping:
    - COWORK_LLM_PROVIDER → llm.provider
    - COWORK_LLM_MODEL → llm.model
    - OLLAMA_BASE_URL → providers.ollama.base_url
    - OPENAI_API_KEY → providers.openai.api_key
    - ANTHROPIC_API_KEY → providers.anthropic.api_key
    """
    # Load defaults
    default_path = Path(__file__).parent / "default_config.yaml"
    with open(default_path) as f:
        data = yaml.safe_load(f)

    # Overlay user config
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_data = yaml.safe_load(f) or {}
        data = _deep_merge(data, user_data)

    # Apply environment variable overrides
    env_mappings = {
        "COWORK_LLM_PROVIDER": "llm.provider",
        "COWORK_LLM_MODEL": "llm.model",
        "COWORK_LLM_TEMPERATURE": "llm.temperature",
        "OLLAMA_BASE_URL": "providers.ollama.base_url",
        "OLLAMA_MODEL": "llm.model",
        "OPENAI_API_KEY": "providers.openai.api_key",
        "OPENAI_BASE_URL": "providers.openai.base_url",
        "ANTHROPIC_API_KEY": "providers.anthropic.api_key",
        "COWORK_WORKSPACE": "agent.workspace_dir",
    }

    config = Config(data)
    for env_key, config_key in env_mappings.items():
        env_val = os.getenv(env_key)
        if env_val is not None:
            # Convert numeric strings
            if config_key.endswith("temperature"):
                env_val = float(env_val)
            config.set(config_key, env_val)

    return config


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay dict into base dict."""
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
