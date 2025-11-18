"""Data loading utilities (stubs)."""
from pathlib import Path
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV file from path and return DataFrame (placeholder)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(p)