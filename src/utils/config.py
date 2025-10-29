"""Configuration loader utility"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

class Config:
    """Global configuration manager"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.config_dir = self.project_root / "config"

        # Load YAML configs
        self.models = self._load_yaml("models.yaml")
        self.trading_rules = self._load_yaml("trading_rules.yaml")

        # Environment variables
        self.env = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "hyperliquid_api_url": os.getenv("HYPERLIQUID_API_URL", "https://api.hyperliquid.xyz"),
            "starting_capital": float(os.getenv("STARTING_CAPITAL", "10000")),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        filepath = self.config_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r') as f:
            return yaml.safe_load(f)

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        models = self.models.get("models", {})
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        return models[model_name]

    def get_trading_symbols(self) -> list[str]:
        """Get list of trading symbols"""
        return self.trading_rules.get("trading_rules", {}).get("symbols", [])

    def get_leverage_range(self) -> tuple[int, int]:
        """Get min and max leverage"""
        leverage = self.trading_rules.get("trading_rules", {}).get("leverage", {})
        return (leverage.get("min", 10), leverage.get("max", 20))


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file

    Args:
        filepath: Path to YAML file (relative to project root or absolute)

    Returns:
        Parsed YAML configuration as dictionary
    """
    path = Path(filepath)

    # If relative path, make it relative to project root
    if not path.is_absolute():
        project_root = Path(__file__).parent.parent.parent
        path = project_root / filepath

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        return yaml.safe_load(f)


# Global config instance
config = Config()
