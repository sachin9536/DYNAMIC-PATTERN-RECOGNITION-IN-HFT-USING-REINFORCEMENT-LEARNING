"""Feature engineering module for high-frequency trading data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute base features from order book and trade data.
    
    Args:
        df: DataFrame with columns: timestamp, bid_price, ask_price, bid_size, 
            ask_size, trade_price, trade_volume
    
    Returns:
        DataFrame with additional feature columns
    """
    logger.info("Computing base features...")
    
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Ensure timestamp is datetime
    if 'timestamp' in result_df.columns:
        result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
    
    # Basic price features
    if 'mid_price' not in result_df.columns:
        result_df['mid_price'] = (result_df['bid_price'] + result_df['ask_price']) / 2
    
    if 'spread' not in result_df.columns:
        result_df['spread'] = result_df['ask_price'] - result_df['bid_price']
    
    # Log returns (vectorized)
    result_df['log_return'] = np.log(result_df['mid_price'] / result_df['mid_price'].shift(1))
    
    # Order imbalance (handle division by zero)
    total_size = result_df['bid_size'] + result_df['ask_size']
    result_df['order_imbalance'] = np.where(
        total_size > 0,
        (result_df['bid_size'] - result_df['ask_size']) / total_size,
        0.0
    )
    
    # Trade intensity (sum of trade volume per row - already available)
    result_df['trade_intensity'] = result_df['trade_volume'].fillna(0)
    
    # VWAP per row (using trade_price and trade_volume)
    # For single-row VWAP, it's just the trade_price when volume > 0
    result_df['vwap'] = np.where(
        result_df['trade_volume'] > 0,
        result_df['trade_price'],
        result_df['mid_price']  # fallback to mid_price
    )
    
    # Elapsed time in milliseconds from first timestamp
    if 'timestamp' in result_df.columns:
        first_timestamp = result_df['timestamp'].iloc[0]
        result_df['elapsed_ms'] = (result_df['timestamp'] - first_timestamp).dt.total_seconds() * 1000
    else:
        result_df['elapsed_ms'] = np.arange(len(result_df)) * 50  # assume 50ms intervals
    
    # Rolling volatility (window=10)
    window = 10
    result_df[f'rolling_vol_{window}'] = result_df['log_return'].rolling(
        window=window, min_periods=1
    ).std()
    
    # Cancellation ratio placeholder (stub implementation)
    result_df['cancellation_ratio'] = 0.0  # placeholder
    logger.warning("cancellation_ratio set to placeholder value 0.0")
    
    # Fill NaN values in log_return (first row)
    result_df['log_return'] = result_df['log_return'].fillna(0.0)
    
    logger.info(f"Base features computed. Added columns: {[col for col in result_df.columns if col not in df.columns]}")
    
    return result_df


def normalize_features(
    df: pd.DataFrame, 
    cols: List[str], 
    method: str = 'zscore'
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Normalize specified features using z-score or min-max scaling.
    
    Args:
        df: Input DataFrame
        cols: List of column names to normalize
        method: 'zscore' or 'minmax'
    
    Returns:
        Tuple of (normalized_df, scalers_dict)
    """
    logger.info(f"Normalizing features: {cols} using method: {method}")
    
    result_df = df.copy()
    scalers = {}
    
    for col in cols:
        if col not in result_df.columns:
            logger.warning(f"Column {col} not found in DataFrame, skipping")
            continue
        
        if method == 'zscore':
            mean_val = result_df[col].mean()
            std_val = result_df[col].std()
            
            if std_val > 0:
                result_df[f'{col}_z'] = (result_df[col] - mean_val) / std_val
                scalers[col] = {'mean': mean_val, 'std': std_val, 'method': 'zscore'}
                logger.info(f"Z-score normalized {col}: mean={mean_val:.6f}, std={std_val:.6f}")
            else:
                result_df[f'{col}_z'] = 0.0
                scalers[col] = {'mean': mean_val, 'std': 0.0, 'method': 'zscore'}
                logger.warning(f"Column {col} has zero std, setting normalized values to 0")
        
        elif method == 'minmax':
            min_val = result_df[col].min()
            max_val = result_df[col].max()
            
            if max_val > min_val:
                result_df[f'{col}_minmax'] = (result_df[col] - min_val) / (max_val - min_val)
                scalers[col] = {'min': min_val, 'max': max_val, 'method': 'minmax'}
                logger.info(f"MinMax normalized {col}: min={min_val:.6f}, max={max_val:.6f}")
            else:
                result_df[f'{col}_minmax'] = 0.0
                scalers[col] = {'min': min_val, 'max': max_val, 'method': 'minmax'}
                logger.warning(f"Column {col} has zero range, setting normalized values to 0")
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return result_df, scalers


def inverse_normalize_features(
    df: pd.DataFrame, 
    scalers: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Inverse transform normalized features back to original scale.
    
    Args:
        df: DataFrame with normalized features
        scalers: Scalers dictionary from normalize_features
    
    Returns:
        DataFrame with inverse-transformed features
    """
    result_df = df.copy()
    
    for col, scaler_info in scalers.items():
        method = scaler_info['method']
        
        if method == 'zscore':
            norm_col = f'{col}_z'
            if norm_col in result_df.columns:
                mean_val = scaler_info['mean']
                std_val = scaler_info['std']
                result_df[f'{col}_inv'] = result_df[norm_col] * std_val + mean_val
        
        elif method == 'minmax':
            norm_col = f'{col}_minmax'
            if norm_col in result_df.columns:
                min_val = scaler_info['min']
                max_val = scaler_info['max']
                result_df[f'{col}_inv'] = result_df[norm_col] * (max_val - min_val) + min_val
    
    return result_df


def select_feature_columns(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Select specific feature columns from DataFrame.
    
    Args:
        df: Input DataFrame
        feature_cols: List of column names to select. If None, selects all _z columns
    
    Returns:
        DataFrame with selected feature columns
    """
    if feature_cols is None:
        # Auto-select normalized columns
        feature_cols = [col for col in df.columns if col.endswith('_z')]
        logger.info(f"Auto-selected feature columns: {feature_cols}")
    
    # Validate columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing feature columns: {missing_cols}")
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    return df[feature_cols].copy()


if __name__ == "__main__":
    # Test with sample data
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    try:
        # Load sample preprocessed data
        sample_path = "data/processed/sample_preprocessed.csv"
        df = pd.read_csv(sample_path)
        
        print(f"Loaded sample data: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Compute base features
        df_features = compute_base_features(df)
        print(f"\nAfter base features: {df_features.shape}")
        print(f"New columns: {[col for col in df_features.columns if col not in df.columns]}")
        
        # Normalize features
        feature_cols = ['mid_price', 'spread', 'order_imbalance', 'trade_intensity', 'rolling_vol_10']
        df_norm, scalers = normalize_features(df_features, feature_cols)
        print(f"\nAfter normalization: {df_norm.shape}")
        print(f"Scalers: {list(scalers.keys())}")
        
        # Show sample of results
        print(f"\nSample results:")
        print(df_norm[['mid_price', 'mid_price_z', 'order_imbalance', 'order_imbalance_z']].head())
        
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()