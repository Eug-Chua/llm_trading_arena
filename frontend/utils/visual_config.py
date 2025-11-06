"""
Visual Configuration Loader

Centralized configuration for frontend colors, chart settings, and UI defaults.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

# Cache the config to avoid repeated file reads
_config_cache = None


def load_visual_config() -> Dict[str, Any]:
    """
    Load visual configuration from YAML file.

    Returns:
        Dictionary containing all visual configuration settings
    """
    global _config_cache

    if _config_cache is not None:
        return _config_cache

    config_path = Path(__file__).parent.parent / "config" / "visual_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Visual config not found at {config_path}")

    with open(config_path, 'r') as f:
        _config_cache = yaml.safe_load(f)

    return _config_cache


def get_model_color(model_name: str) -> str:
    """
    Get consistent color for a model.

    Args:
        model_name: Model identifier (e.g., 'anthropic', 'openai', 'Anthropic', 'OpenAI')

    Returns:
        Hex color code
    """
    config = load_visual_config()
    model_colors = config.get('model_colors', {})

    # Normalize to lowercase and remove spaces for lookup
    normalized = model_name.lower().replace(' ', '-')

    # Return mapped color or default to gray
    return model_colors.get(normalized, '#808080')


def get_coin_color(coin_symbol: str) -> str:
    """
    Get consistent color for a coin.

    Args:
        coin_symbol: Coin symbol (e.g., 'BTC', 'ETH')

    Returns:
        Hex color code
    """
    config = load_visual_config()
    coin_colors = config.get('coin_colors', {})

    return coin_colors.get(coin_symbol.upper(), '#888888')


def get_chart_height(chart_type: str) -> int:
    """
    Get standard height for a chart type.

    Args:
        chart_type: Chart type (e.g., 'performance_chart', 'candlestick_chart')

    Returns:
        Height in pixels
    """
    config = load_visual_config()
    dimensions = config.get('chart_dimensions', {})

    return dimensions.get(f'{chart_type}_height', 600)


def get_indicator_colors() -> Dict[str, str]:
    """Get all indicator colors."""
    config = load_visual_config()
    return config.get('indicator_colors', {})


def get_candlestick_colors() -> Dict[str, str]:
    """Get candlestick colors."""
    config = load_visual_config()
    return config.get('candlestick_colors', {})


def get_trade_marker_settings() -> Dict[str, Any]:
    """Get trade marker settings."""
    config = load_visual_config()
    return config.get('trade_markers', {})


def get_ui_defaults() -> Dict[str, Any]:
    """Get UI default settings."""
    config = load_visual_config()
    return config.get('ui_defaults', {})
