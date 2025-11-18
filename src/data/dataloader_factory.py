"""DataLoader factory and utilities for market sequence data."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

try:
    from src.data.sequence_builder import MarketSequenceDataset, load_sequences, save_sequences, split_sequences
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def get_dataloader(
    seq_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    """
    Create a DataLoader from saved sequences.
    
    Args:
        seq_path: Path to .npz file with sequences
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
    
    Returns:
        PyTorch DataLoader
    """
    logger.info(f"Creating DataLoader from {seq_path}")
    
    # Load sequences
    sequences, targets, metadata = load_sequences(seq_path)
    
    # Create dataset
    dataset = MarketSequenceDataset(sequences, targets, metadata)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"Created DataLoader: {len(dataset)} samples, batch_size={batch_size}")
    
    return dataloader


def create_train_val_test_loaders(
    sequences: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
    save_dir: Optional[str] = None,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/validation/test DataLoaders from sequences.
    
    Args:
        sequences: Input sequences array
        targets: Input targets array
        batch_size: Batch size for all loaders
        train_frac: Fraction for training set
        val_frac: Fraction for validation set
        seed: Random seed for splitting
        save_dir: Optional directory to save split datasets
        num_workers: Number of worker processes
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info("Creating train/val/test DataLoaders")
    
    # Split sequences
    train_data, val_data, test_data = split_sequences(
        sequences, targets, train_frac, val_frac, seed
    )
    
    # Optionally save split datasets
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_sequences(train_data[0], train_data[1], str(save_dir / "train.npz"))
        save_sequences(val_data[0], val_data[1], str(save_dir / "val.npz"))
        save_sequences(test_data[0], test_data[1], str(save_dir / "test.npz"))
        
        logger.info(f"Saved split datasets to {save_dir}")
    
    # Create datasets
    train_dataset = MarketSequenceDataset(train_data[0], train_data[1])
    val_dataset = MarketSequenceDataset(val_data[0], val_data[1])
    test_dataset = MarketSequenceDataset(test_data[0], test_data[1])
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    logger.info(f"Created DataLoaders: train={len(train_dataset)}, "
               f"val={len(val_dataset)}, test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def get_dataloader_from_config(
    config: Dict[str, Any],
    data_path: str,
    split: str = 'train'
) -> DataLoader:
    """
    Create DataLoader based on configuration.
    
    Args:
        config: Configuration dictionary
        data_path: Path to data directory or specific file
        split: 'train', 'val', or 'test'
    
    Returns:
        DataLoader for specified split
    """
    # Extract config parameters
    batch_size = config.get('training', {}).get('batch_size', 32)
    num_workers = config.get('dataloader', {}).get('num_workers', 0)
    
    # Determine file path
    data_path = Path(data_path)
    if data_path.is_dir():
        seq_path = data_path / f"{split}.npz"
    else:
        seq_path = data_path
    
    # Create DataLoader
    shuffle = (split == 'train')
    
    return get_dataloader(
        str(seq_path),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def collate_sequences(batch):
    """
    Custom collate function for sequence batches.
    
    Args:
        batch: List of (sequence, target, metadata) tuples
    
    Returns:
        Batched tensors and metadata
    """
    sequences, targets, metadata_list = zip(*batch)
    
    # Stack sequences and targets
    sequences_batch = torch.stack(sequences)
    targets_batch = torch.stack(targets)
    
    # Combine metadata
    batch_metadata = {}
    if metadata_list:
        # Get all keys from first metadata dict
        keys = metadata_list[0].keys()
        for key in keys:
            values = [meta.get(key) for meta in metadata_list]
            batch_metadata[key] = values
    
    return sequences_batch, targets_batch, batch_metadata


def get_sample_batch(dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Get a single sample batch from DataLoader for testing.
    
    Args:
        dataloader: PyTorch DataLoader
    
    Returns:
        Tuple of (sequences_batch, targets_batch, metadata_batch)
    """
    for batch in dataloader:
        return batch
    
    raise ValueError("DataLoader is empty")


if __name__ == "__main__":
    # Test DataLoader creation
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    try:
        # Create sample sequences
        n_samples = 100
        seq_len = 20
        n_features = 5
        
        np.random.seed(42)
        sequences = np.random.randn(n_samples, seq_len, n_features)
        targets = np.random.randn(n_samples)
        
        print(f"Sample sequences shape: {sequences.shape}")
        print(f"Sample targets shape: {targets.shape}")
        
        # Create train/val/test loaders
        train_loader, val_loader, test_loader = create_train_val_test_loaders(
            sequences, targets, batch_size=16
        )
        
        print(f"Train loader: {len(train_loader.dataset)} samples")
        print(f"Val loader: {len(val_loader.dataset)} samples")
        print(f"Test loader: {len(test_loader.dataset)} samples")
        
        # Test sample batch
        seq_batch, target_batch, metadata_batch = get_sample_batch(train_loader)
        print(f"Sample batch shapes: sequences={seq_batch.shape}, targets={target_batch.shape}")
        print(f"Batch metadata keys: {list(metadata_batch.keys())}")
        
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()