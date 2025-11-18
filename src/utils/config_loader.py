"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration data
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config or {}