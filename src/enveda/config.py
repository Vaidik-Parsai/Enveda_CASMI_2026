"""Configuration loading and validation."""

from pathlib import Path
from typing import Any, Dict
import yaml


class ConfigDict(dict):
    """Dictionary that supports attribute-style access."""

    def __getattr__(self, name: str) -> Any:
        try:
            val = self[name]
            if isinstance(val, dict) and not isinstance(val, ConfigDict):
                val = ConfigDict(val)
                self[name] = val
            return val
        except KeyError:
            raise AttributeError(f"No configuration key '{name}' found.")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(config_path: str = "configs/default.yaml") -> ConfigDict:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML file.
        
    Returns:
        ConfigDict containing configuration parameters.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ConfigDict(data)
