"""Load real RL observations for explainability."""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import random

try:
    from src.data.sequence_builder import load_sequences
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class RLObservationLoader:
    """Load real observations from RL training/testing data."""
    
    def __init__(self, sequence_paths: Optional[list] = None):
        """
        Initialize observation loader.
        
        Args:
            sequence_paths: List of paths to sequence files. If None, uses defaults.
        """
        if sequence_paths is None:
            sequence_paths = [
                "PR_project/data/processed/sequences/sample_sequences.npz",
                "PR_project/data/processed/sequences/synthetic_sequences.npz",
                "PR_project/data/processed/sequences/final_test.npz",
                "PR_project/data/processed/sequences/clean_test.npz",
                "PR_project/data/processed/sequences/my_sequences.npz",
                "data/processed/sequences/sample_sequences.npz",
                "data/processed/sequences/synthetic_sequences.npz",
                "data/processed/sequences/final_test.npz",
                "data/processed/sequences/clean_test.npz",
                "data/processed/sequences/my_sequences.npz"
            ]
        
        self.sequence_paths = sequence_paths
        self.sequences = None
        self.targets = None
        self.metadata = None
        self.loaded_path = None
        
        # Try to load sequences
        self._load_sequences()
    
    def _load_sequences(self):
        """Load sequences from available files."""
        for path in self.sequence_paths:
            try:
                path_obj = Path(path)
                if path_obj.exists():
                    self.sequences, self.targets, self.metadata = load_sequences(str(path_obj))
                    self.loaded_path = str(path_obj)
                    logger.info(f"Loaded {len(self.sequences)} sequences from {path}")
                    return True
            except Exception as e:
                logger.debug(f"Could not load from {path}: {e}")
        
        logger.warning("No sequence files could be loaded")
        return False
    
    def is_loaded(self) -> bool:
        """Check if sequences are loaded."""
        return self.sequences is not None
    
    def get_observation_shape(self) -> Tuple[int, ...]:
        """Get the shape of observations."""
        if not self.is_loaded():
            return (0,)
        return self.sequences.shape[1:]  # (seq_len, n_features)
    
    def get_num_observations(self) -> int:
        """Get total number of observations."""
        if not self.is_loaded():
            return 0
        return len(self.sequences)
    
    def get_last_observation(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get the last observation from the dataset.
        
        Returns:
            Tuple of (observation, info_dict)
        """
        if not self.is_loaded():
            raise ValueError("No sequences loaded")
        
        idx = len(self.sequences) - 1
        observation = self.sequences[idx].copy()
        
        info = {
            'index': idx,
            'source': 'last',
            'target': self.targets[idx] if self.targets is not None else None,
            'loaded_from': self.loaded_path,
            'shape': observation.shape
        }
        
        return observation, info
    
    def get_random_observation(self, split: str = 'all') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get a random observation from the dataset.
        
        Args:
            split: 'all', 'train', 'val', or 'test'
        
        Returns:
            Tuple of (observation, info_dict)
        """
        if not self.is_loaded():
            raise ValueError("No sequences loaded")
        
        n_total = len(self.sequences)
        
        # Determine index range based on split
        if split == 'train':
            # First 80% for training
            start_idx = 0
            end_idx = int(n_total * 0.8)
        elif split == 'val':
            # Next 10% for validation
            start_idx = int(n_total * 0.8)
            end_idx = int(n_total * 0.9)
        elif split == 'test':
            # Last 10% for testing
            start_idx = int(n_total * 0.9)
            end_idx = n_total
        else:  # 'all'
            start_idx = 0
            end_idx = n_total
        
        if start_idx >= end_idx:
            raise ValueError(f"Invalid split '{split}' for dataset size {n_total}")
        
        # Select random index
        idx = random.randint(start_idx, end_idx - 1)
        observation = self.sequences[idx].copy()
        
        info = {
            'index': idx,
            'source': f'random_{split}',
            'split': split,
            'target': self.targets[idx] if self.targets is not None else None,
            'loaded_from': self.loaded_path,
            'shape': observation.shape
        }
        
        return observation, info
    
    def get_observation_by_index(self, idx: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get observation by index.
        
        Args:
            idx: Index of observation
        
        Returns:
            Tuple of (observation, info_dict)
        """
        if not self.is_loaded():
            raise ValueError("No sequences loaded")
        
        if idx < 0 or idx >= len(self.sequences):
            raise ValueError(f"Index {idx} out of range [0, {len(self.sequences)})")
        
        observation = self.sequences[idx].copy()
        
        info = {
            'index': idx,
            'source': 'by_index',
            'target': self.targets[idx] if self.targets is not None else None,
            'loaded_from': self.loaded_path,
            'shape': observation.shape
        }
        
        return observation, info
    
    def get_train_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get training split (first 80%)."""
        if not self.is_loaded():
            raise ValueError("No sequences loaded")
        
        n_train = int(len(self.sequences) * 0.8)
        return self.sequences[:n_train], self.targets[:n_train]
    
    def get_val_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get validation split (next 10%)."""
        if not self.is_loaded():
            raise ValueError("No sequences loaded")
        
        n_train = int(len(self.sequences) * 0.8)
        n_val = int(len(self.sequences) * 0.9)
        return self.sequences[n_train:n_val], self.targets[n_train:n_val]
    
    def get_test_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test split (last 10%)."""
        if not self.is_loaded():
            raise ValueError("No sequences loaded")
        
        n_test = int(len(self.sequences) * 0.9)
        return self.sequences[n_test:], self.targets[n_test:]
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about loaded sequences."""
        if not self.is_loaded():
            return {
                'loaded': False,
                'message': 'No sequences loaded'
            }
        
        n_total = len(self.sequences)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.9) - n_train
        n_test = n_total - int(n_total * 0.9)
        
        return {
            'loaded': True,
            'loaded_from': self.loaded_path,
            'total_observations': n_total,
            'observation_shape': self.get_observation_shape(),
            'splits': {
                'train': n_train,
                'val': n_val,
                'test': n_test
            },
            'has_targets': self.targets is not None,
            'metadata_keys': list(self.metadata.keys()) if self.metadata else []
        }


def load_latest_observation(split: str = 'all') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load the latest observation from RL training data.
    
    Args:
        split: 'last', 'random_train', 'random_val', 'random_test', or 'random_all'
    
    Returns:
        Tuple of (observation, info_dict)
    """
    loader = RLObservationLoader()
    
    if not loader.is_loaded():
        raise ValueError("Could not load any sequence files")
    
    if split == 'last':
        return loader.get_last_observation()
    elif split.startswith('random_'):
        split_name = split.replace('random_', '')
        return loader.get_random_observation(split_name)
    else:
        return loader.get_last_observation()


if __name__ == "__main__":
    # Test observation loader
    try:
        print("Testing RL Observation Loader...")
        print("=" * 60)
        
        # Initialize loader
        loader = RLObservationLoader()
        
        # Get info
        info = loader.get_info()
        print("\nDataset Info:")
        print(f"  Loaded: {info['loaded']}")
        if info['loaded']:
            print(f"  Source: {info['loaded_from']}")
            print(f"  Total observations: {info['total_observations']}")
            print(f"  Observation shape: {info['observation_shape']}")
            print(f"  Splits: {info['splits']}")
        
        if loader.is_loaded():
            # Test last observation
            print("\n--- Test 1: Last Observation ---")
            obs, obs_info = loader.get_last_observation()
            print(f"  Shape: {obs.shape}")
            print(f"  Index: {obs_info['index']}")
            print(f"  Target: {obs_info['target']}")
            
            # Test random observations
            print("\n--- Test 2: Random Observations ---")
            for split in ['train', 'val', 'test', 'all']:
                obs, obs_info = loader.get_random_observation(split)
                print(f"  {split.upper()}: index={obs_info['index']}, shape={obs.shape}")
            
            # Test by index
            print("\n--- Test 3: By Index ---")
            obs, obs_info = loader.get_observation_by_index(0)
            print(f"  Index 0: shape={obs.shape}, target={obs_info['target']}")
            
            # Test splits
            print("\n--- Test 4: Splits ---")
            train_seq, train_targets = loader.get_train_split()
            val_seq, val_targets = loader.get_val_split()
            test_seq, test_targets = loader.get_test_split()
            print(f"  Train: {len(train_seq)} sequences")
            print(f"  Val: {len(val_seq)} sequences")
            print(f"  Test: {len(test_seq)} sequences")
            
            print("\n✅ All tests passed!")
        else:
            print("\n⚠️  No sequences loaded. Please generate sequences first.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
