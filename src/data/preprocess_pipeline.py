"""Market data preprocessing pipeline for high-frequency trading data."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os

# Add parent directory to path for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class MarketDataPipeline:
    """Pipeline for loading, validating, and preprocessing market data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline with configuration."""
        self.config = config or {}
        logger.info("MarketDataPipeline initialized")
    
    def load_data(self, source_path: str) -> pd.DataFrame:
        """Load market data from CSV file."""
        logger.info(f"Loading data from {source_path}")
        
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {source_path}")
        
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean market data."""
        logger.info("Validating data...")
        
        initial_rows = len(df)
        
        # Check required columns
        required_cols = ['timestamp', 'bid_price', 'ask_price', 'bid_size', 'ask_size']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with missing critical data
        df = df.dropna(subset=required_cols)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicate timestamps
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        # Basic sanity checks
        df = df[df['bid_price'] > 0]
        df = df[df['ask_price'] > 0]
        df = df[df['bid_size'] >= 0]
        df = df[df['ask_size'] >= 0]
        df = df[df['ask_price'] >= df['bid_price']]  # Spread should be non-negative
        
        final_rows = len(df)
        logger.info(f"Validation complete: {initial_rows} -> {final_rows} rows ({initial_rows - final_rows} removed)")
        
        return df
    
    def resample_data(self, df: pd.DataFrame, interval_ms: int = 50) -> pd.DataFrame:
        """Resample tick data into uniform time intervals."""
        logger.info(f"Resampling data to {interval_ms}ms intervals")
        
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have 'timestamp' column for resampling")
        
        # Set timestamp as index
        df_indexed = df.set_index('timestamp')
        
        # Define resampling frequency
        freq = f'{interval_ms}ms'
        
        # Resample with appropriate aggregation
        resampled = df_indexed.resample(freq).agg({
            'bid_price': 'last',
            'ask_price': 'last',
            'bid_size': 'last',
            'ask_size': 'last',
            'trade_price': 'mean',
            'trade_volume': 'sum'
        })
        
        # Forward fill missing values
        resampled = resampled.ffill()
        
        # Reset index to get timestamp back as column
        resampled = resampled.reset_index()
        
        # Add derived features
        resampled['mid_price'] = (resampled['bid_price'] + resampled['ask_price']) / 2
        resampled['spread'] = resampled['ask_price'] - resampled['bid_price']
        
        logger.info(f"Resampling complete: {len(df)} -> {len(resampled)} rows")
        
        return resampled
    
    def normalize_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Normalize specified columns using z-score normalization."""
        logger.info(f"Normalizing columns: {cols}")
        
        df_normalized = df.copy()
        
        for col in cols:
            if col in df_normalized.columns:
                mean_val = df_normalized[col].mean()
                std_val = df_normalized[col].std()
                
                if std_val > 0:
                    df_normalized[f'{col}_normalized'] = (df_normalized[col] - mean_val) / std_val
                    logger.info(f"Normalized {col}: mean={mean_val:.4f}, std={std_val:.4f}")
                else:
                    logger.warning(f"Column {col} has zero std, skipping normalization")
            else:
                logger.warning(f"Column {col} not found in DataFrame")
        
        return df_normalized
    
    def save_processed(self, df: pd.DataFrame, output_path: str) -> None:
        """Save processed data to file."""
        logger.info(f"Saving processed data to {output_path}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file extension
        if output_path.suffix == '.parquet':
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(df)} rows to {output_path}")
    
    def run_full_pipeline(self, source_path: str, output_path: str) -> pd.DataFrame:
        """Run the complete preprocessing pipeline."""
        logger.info("Starting full preprocessing pipeline")
        
        # Load data
        df = self.load_data(source_path)
        
        # Validate and clean
        df = self.validate_data(df)
        
        # Resample into time bins
        interval_ms = self.config.get('preprocessing', {}).get('interval_ms', 50)
        df = self.resample_data(df, interval_ms)
        
        # Normalize features
        normalize_cols = self.config.get('preprocessing', {}).get('normalize_columns', [
            'bid_price', 'ask_price', 'bid_size', 'ask_size', 'trade_volume'
        ])
        df = self.normalize_features(df, normalize_cols)
        
        # Save processed data
        self.save_processed(df, output_path)
        
        logger.info("Full preprocessing pipeline completed successfully")
        return df


if __name__ == "__main__":
    # Example usage
    pipeline = MarketDataPipeline()
    
    input_path = "data/raw/sample_data.csv"
    output_path = "data/processed/sample_preprocessed.csv"
    
    try:
        result_df = pipeline.run_full_pipeline(input_path, output_path)
        print(f"Pipeline completed successfully!")
        print(f"Processed {len(result_df)} rows")
        print(f"Output saved to: {output_path}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")