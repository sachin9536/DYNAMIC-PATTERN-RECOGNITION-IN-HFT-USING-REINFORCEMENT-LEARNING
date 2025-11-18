"""Sequence building and PyTorch dataset for time series data."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def build_sequences(
    df: pd.DataFrame,
    seq_len: int = 100,
    step: int = 1,
    mode: str = 'sliding',
    feature_cols: Optional[List[str]] = None
) -> np.ndarray:
    """
    Build sequences from time series data.
    
    Args:
        df: Input DataFrame with features
        seq_len: Length of each sequence
        step: Step size between sequences (for sliding mode)
        mode: 'sliding' or 'chunk'
        feature_cols: List of feature columns to use. If None, uses all numeric columns
    
    Returns:
        numpy array of shape (N_sequences, seq_len, n_features)
    """
    logger.info(f"Building sequences: seq_len={seq_len}, step={step}, mode={mode}")
    
    # Select feature columns
    if feature_cols is None:
        # Use all numeric columns except timestamp
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'elapsed_ms' in feature_cols:
            feature_cols.remove('elapsed_ms')  # Remove time index
        logger.info(f"Auto-selected {len(feature_cols)} feature columns")
    
    # Validate feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    # Extract feature matrix
    feature_matrix = df[feature_cols].values
    n_samples, n_features = feature_matrix.shape
    
    logger.info(f"Feature matrix shape: {feature_matrix.shape}")
    
    if n_samples < seq_len:
        raise ValueError(f"Not enough samples ({n_samples}) for sequence length ({seq_len})")
    
    sequences = []
    
    if mode == 'sliding':
        # Sliding window approach
        for i in range(0, n_samples - seq_len + 1, step):
            sequence = feature_matrix[i:i + seq_len]
            sequences.append(sequence)
    
    elif mode == 'chunk':
        # Non-overlapping chunks
        for i in range(0, n_samples - seq_len + 1, seq_len):
            sequence = feature_matrix[i:i + seq_len]
            sequences.append(sequence)
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'sliding' or 'chunk'")
    
    sequences_array = np.array(sequences)
    logger.info(f"Built {len(sequences)} sequences of shape {sequences_array.shape}")
    
    return sequences_array


def build_targets(
    df: pd.DataFrame,
    horizon: int = 1,
    target_col: str = 'mid_price',
    threshold: float = 0.0005,
    target_type: str = 'return'
) -> np.ndarray:
    """
    Build target labels for sequences.
    
    Args:
        df: Input DataFrame
        horizon: Number of steps ahead to predict
        target_col: Column to use for target computation
        threshold: Threshold for binary classification
        target_type: 'return' for regression, 'binary' for classification
    
    Returns:
        numpy array of targets aligned with sequences
    """
    logger.info(f"Building targets: horizon={horizon}, target_col={target_col}, type={target_type}")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in DataFrame")
    
    values = df[target_col].values
    n_samples = len(values)
    
    targets = []
    
    for i in range(n_samples - horizon):
        current_val = values[i]
        future_val = values[i + horizon]
        
        if target_type == 'return':
            # Future return
            if current_val != 0:
                target = (future_val / current_val) - 1
            else:
                target = 0.0
        
        elif target_type == 'binary':
            # Binary classification based on threshold
            if current_val != 0:
                future_return = (future_val / current_val) - 1
                target = 1.0 if future_return > threshold else 0.0
            else:
                target = 0.0
        
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        targets.append(target)
    
    # Pad with last value to match sequence count
    while len(targets) < n_samples:
        targets.append(targets[-1] if targets else 0.0)
    
    targets_array = np.array(targets)
    logger.info(f"Built {len(targets)} targets")
    
    return targets_array


class MarketSequenceDataset(Dataset):
    """PyTorch Dataset for market sequences."""
    
    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        metadata: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Initialize dataset.
        
        Args:
            sequences: Array of shape (N, seq_len, n_features)
            targets: Array of shape (N,)
            metadata: Optional dictionary with additional info
        """
        self.sequences = sequences
        self.targets = targets
        self.metadata = metadata or {}
        
        # Validate shapes
        if len(sequences) != len(targets):
            raise ValueError(f"Sequences ({len(sequences)}) and targets ({len(targets)}) length mismatch")
        
        logger.info(f"Dataset initialized: {len(sequences)} samples, "
                   f"sequence shape: {sequences.shape[1:]}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Get a single sample."""
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor([self.targets[idx]])
        
        # Extract metadata for this index
        sample_metadata = {}
        for key, values in self.metadata.items():
            if isinstance(values, np.ndarray) and len(values) > idx:
                sample_metadata[key] = values[idx]
            else:
                sample_metadata[key] = values
        
        sample_metadata['index'] = idx
        
        return sequence, target, sample_metadata


def save_sequences(
    sequences: np.ndarray,
    targets: np.ndarray,
    out_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save sequences and targets to .npz file.
    
    Args:
        sequences: Sequences array
        targets: Targets array
        out_path: Output file path
        metadata: Optional metadata dictionary
    """
    logger.info(f"Saving sequences to {out_path}")
    
    # Create output directory
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving
    save_dict = {
        'sequences': sequences,
        'targets': targets
    }
    
    if metadata:
        for key, value in metadata.items():
            save_dict[f'meta_{key}'] = value
    
    np.savez_compressed(out_path, **save_dict)
    logger.info(f"Saved {len(sequences)} sequences to {out_path}")


def load_sequences(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load sequences and targets from .npz file.
    
    Args:
        path: Path to .npz file
    
    Returns:
        Tuple of (sequences, targets, metadata)
    """
    logger.info(f"Loading sequences from {path}")
    
    if not Path(path).exists():
        raise FileNotFoundError(f"Sequence file not found: {path}")
    
    data = np.load(path)
    
    sequences = data['sequences']
    targets = data['targets']
    
    # Extract metadata
    metadata = {}
    for key in data.keys():
        if key.startswith('meta_'):
            metadata[key[5:]] = data[key]  # Remove 'meta_' prefix
    
    logger.info(f"Loaded {len(sequences)} sequences from {path}")
    
    return sequences, targets, metadata


def split_sequences(
    sequences: np.ndarray,
    targets: np.ndarray,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Split sequences into train/validation/test sets.
    
    Args:
        sequences: Input sequences
        targets: Input targets
        train_frac: Fraction for training
        val_frac: Fraction for validation (test gets remainder)
        seed: Random seed
    
    Returns:
        Tuple of ((train_seq, train_targets), (val_seq, val_targets), (test_seq, test_targets))
    """
    np.random.seed(seed)
    
    n_samples = len(sequences)
    indices = np.random.permutation(n_samples)
    
    train_end = int(n_samples * train_frac)
    val_end = int(n_samples * (train_frac + val_frac))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    train_data = (sequences[train_idx], targets[train_idx])
    val_data = (sequences[val_idx], targets[val_idx])
    test_data = (sequences[test_idx], targets[test_idx])
    
    logger.info(f"Split sequences: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    # Test sequence building
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    try:
        # Create sample data
        n_samples = 200
        n_features = 5
        
        # Generate synthetic time series data
        np.random.seed(42)
        data = np.random.randn(n_samples, n_features).cumsum(axis=0)
        
        df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
        df['mid_price'] = 100 + np.random.randn(n_samples).cumsum() * 0.1
        
        print(f"Sample data shape: {df.shape}")
        
        # Build sequences
        sequences = build_sequences(df, seq_len=50, step=10, mode='sliding')
        print(f"Sequences shape: {sequences.shape}")
        
        # Build targets
        targets = build_targets(df, horizon=1, target_col='mid_price')
        print(f"Targets shape: {targets.shape}")
        
        # Create dataset
        dataset = MarketSequenceDataset(sequences, targets[:len(sequences)])
        print(f"Dataset length: {len(dataset)}")
        
        # Test sample
        seq, target, metadata = dataset[0]
        print(f"Sample shapes: sequence={seq.shape}, target={target.shape}")
        print(f"Sample metadata: {metadata}")
        
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()