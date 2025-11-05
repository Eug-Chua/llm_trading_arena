"""
Shared utilities for loading and processing checkpoint files.
Used across multiple pages to avoid code duplication.
"""

import pickle
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Union


def natural_sort_key(item: Union[str, Path]) -> List:
    """
    Sort key function that handles numbers naturally (trial1, trial2, ..., trial10).

    This prevents alphabetical sorting where trial10 comes before trial2.

    Args:
        item: Filename string or Path object

    Returns:
        List of alternating strings and integers for natural comparison

    Example:
        >>> files = ['trial1.pkl', 'trial10.pkl', 'trial2.pkl']
        >>> sorted(files, key=natural_sort_key)
        ['trial1.pkl', 'trial2.pkl', 'trial10.pkl']
    """
    filename = str(item.name if isinstance(item, Path) else item)
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', filename)]


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load checkpoint data from pickle file.

    Args:
        checkpoint_path: Path to the checkpoint .pkl file

    Returns:
        Dictionary containing checkpoint data
    """
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)


def load_checkpoint_json(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load checkpoint data from JSON file.

    Args:
        checkpoint_path: Path to the checkpoint .json file

    Returns:
        Dictionary containing checkpoint data
    """
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def load_reasoning(reasoning_path: str) -> Dict[str, Any]:
    """
    Load LLM reasoning data from JSON file.

    Args:
        reasoning_path: Path to the reasoning .json file

    Returns:
        Dictionary containing reasoning data, or empty dict if file not found
    """
    try:
        with open(reasoning_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def get_checkpoint_paths(results_dir: str = "results") -> list[str]:
    """
    Get all checkpoint file paths from the results directory (recursively).

    Supports both old flat structure (results/checkpoints/*.pkl) and new nested structure
    (results/{model}/temp{temp}/*.pkl).

    Args:
        results_dir: Root directory containing checkpoint files (default: results)

    Returns:
        List of checkpoint file paths
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        return []

    # Recursively find all .pkl files
    return [str(p) for p in results_path.rglob("*.pkl")]


def group_checkpoints_by_model(checkpoint_paths: list[str]) -> Dict[str, list[str]]:
    """
    Group checkpoint paths by model name.

    Args:
        checkpoint_paths: List of checkpoint file paths

    Returns:
        Dictionary mapping model names to lists of checkpoint paths
    """
    grouped = {}

    for path in checkpoint_paths:
        try:
            checkpoint = load_checkpoint(path)
            model_name = checkpoint.get('metadata', {}).get('model', 'unknown')

            if model_name not in grouped:
                grouped[model_name] = []

            grouped[model_name].append(path)
        except Exception:
            continue

    return grouped
