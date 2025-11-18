"""Caching utilities for explainability results."""

import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class ExplanationCache:
    """Cache manager for explanation results."""
    
    def __init__(self, cache_dir: str = "artifacts/explanations"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def store(self, key: str, data: Dict[str, Any]) -> bool:
        """Store explanation data with given key."""
        try:
            filepath = self.cache_dir / f"{key}.json"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=_json_serializer)
            return True
        except Exception as e:
            logger.error(f"Failed to store cache entry {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve explanation data by key."""
        try:
            filepath = self.cache_dir / f"{key}.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve cache entry {key}: {e}")
            return None
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        filepath = self.cache_dir / f"{key}.json"
        return filepath.exists()
    
    def clear(self) -> int:
        """Clear all cache entries."""
        return clear_cache(str(self.cache_dir))
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        df = list_cached_explanations(str(self.cache_dir))
        return {
            'total_entries': len(df),
            'methods': df['method'].unique().tolist() if not df.empty else [],
            'cache_dir': str(self.cache_dir),
            'size_mb': sum(f.stat().st_size for f in self.cache_dir.glob('*.json')) / (1024 * 1024)
        }


def make_input_hash(model_id: str, method: str, observation: np.ndarray) -> str:
    """
    Create deterministic hash for caching explanations.
    
    Args:
        model_id: Unique model identifier
        method: Explanation method
        observation: Input observation
    
    Returns:
        Hash string for caching
    """
    # Create content string for hashing
    content_parts = [
        model_id,
        method,
        str(observation.shape),
        str(observation.dtype)
    ]
    
    # Add sample of observation values (to handle similar but not identical inputs)
    if observation.size > 0:
        # Use a few representative values
        flat_obs = observation.flatten()
        sample_indices = np.linspace(0, len(flat_obs)-1, min(10, len(flat_obs)), dtype=int)
        sample_values = flat_obs[sample_indices]
        content_parts.append(str(sample_values.tolist()))
    
    content_string = "|".join(content_parts)
    
    # Create hash
    hash_obj = hashlib.sha256(content_string.encode())
    return hash_obj.hexdigest()[:16]  # Use first 16 characters


def save_explanation(
    explanation: Dict[str, Any],
    cache_dir: str,
    hash_key: Optional[str] = None
) -> str:
    """
    Save explanation to cache directory.
    
    Args:
        explanation: Explanation dictionary
        cache_dir: Cache directory path
        hash_key: Optional hash key (will generate if not provided)
    
    Returns:
        Path to saved file
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate hash key if not provided
    if hash_key is None:
        content_str = json.dumps(explanation, sort_keys=True, default=str)
        hash_key = hashlib.md5(content_str.encode()).hexdigest()[:16]
    
    # Create filename
    method = explanation.get('method', 'unknown')
    filename = f"{method}_{hash_key}.json"
    filepath = cache_dir / filename
    
    # Save to JSON
    try:
        with open(filepath, 'w') as f:
            json.dump(explanation, f, indent=2, default=_json_serializer)
        
        logger.debug(f"Saved explanation to {filepath}")
        return str(filepath)
    
    except Exception as e:
        logger.error(f"Failed to save explanation: {e}")
        return ""


def load_explanation(hash_key: str, cache_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load explanation from cache.
    
    Args:
        hash_key: Hash key for the explanation
        cache_dir: Cache directory path
    
    Returns:
        Explanation dictionary or None if not found
    """
    cache_dir = Path(cache_dir)
    
    # Look for files with this hash key
    pattern = f"*_{hash_key}.json"
    matching_files = list(cache_dir.glob(pattern))
    
    if not matching_files:
        logger.debug(f"No cached explanation found for hash {hash_key}")
        return None
    
    # Load the first matching file
    filepath = matching_files[0]
    
    try:
        with open(filepath, 'r') as f:
            explanation = json.load(f)
        
        logger.debug(f"Loaded explanation from {filepath}")
        return explanation
    
    except Exception as e:
        logger.error(f"Failed to load explanation from {filepath}: {e}")
        return None


def _json_serializer(obj):
    """Custom JSON serializer for numpy arrays and other objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return str(obj)


def list_cached_explanations(cache_dir: str) -> pd.DataFrame:
    """
    List all cached explanations.
    
    Args:
        cache_dir: Cache directory path
    
    Returns:
        DataFrame with explanation metadata
    """
    cache_dir = Path(cache_dir)
    
    if not cache_dir.exists():
        return pd.DataFrame(columns=['filename', 'method', 'hash_key', 'timestamp', 'model_id'])
    
    explanations = []
    
    for filepath in cache_dir.glob("*.json"):
        try:
            with open(filepath, 'r') as f:
                explanation = json.load(f)
            
            explanations.append({
                'filename': filepath.name,
                'method': explanation.get('method', 'unknown'),
                'hash_key': explanation.get('input_hash', 'unknown'),
                'timestamp': explanation.get('timestamp', 'unknown'),
                'model_id': explanation.get('model_id', 'unknown'),
                'has_error': 'error' in explanation
            })
        
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")
    
    return pd.DataFrame(explanations)


def clear_cache(cache_dir: str, method: Optional[str] = None) -> int:
    """
    Clear cached explanations.
    
    Args:
        cache_dir: Cache directory path
        method: Optional method to clear (if None, clears all)
    
    Returns:
        Number of files removed
    """
    cache_dir = Path(cache_dir)
    
    if not cache_dir.exists():
        return 0
    
    if method:
        pattern = f"{method}_*.json"
    else:
        pattern = "*.json"
    
    files_to_remove = list(cache_dir.glob(pattern))
    
    for filepath in files_to_remove:
        try:
            filepath.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove {filepath}: {e}")
    
    removed_count = len(files_to_remove)
    logger.info(f"Removed {removed_count} cached explanations")
    
    return removed_count


if __name__ == "__main__":
    # Test caching functionality
    try:
        print("Testing explanation caching...")
        
        # Create test explanation
        test_explanation = {
            'method': 'test',
            'model_id': 'test_model',
            'feature_names': ['f1', 'f2', 'f3'],
            'values': [0.1, 0.2, 0.3],
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Test saving
        cache_dir = "artifacts/explanations/test"
        saved_path = save_explanation(test_explanation, cache_dir)
        print(f"Saved test explanation to: {saved_path}")
        
        # Test hash generation
        model_id = "test_model"
        method = "test"
        observation = np.random.randn(10, 5)
        hash_key = make_input_hash(model_id, method, observation)
        print(f"Generated hash key: {hash_key}")
        
        # Test loading
        loaded = load_explanation(hash_key, cache_dir)
        if loaded:
            print("✅ Successfully loaded from cache")
        else:
            print("❌ Failed to load from cache")
        
        # Test listing
        df = list_cached_explanations(cache_dir)
        print(f"Found {len(df)} cached explanations")
        if not df.empty:
            print(df.head())
        
        # Test clearing
        removed = clear_cache(cache_dir)
        print(f"Removed {removed} cached files")
        
        print("Caching test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()