#!/usr/bin/env python3
"""Command-line interface for running data preprocessing pipeline."""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess_pipeline import MarketDataPipeline
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Run market data preprocessing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input data file (CSV)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output processed data file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Initialize pipeline
        pipeline = MarketDataPipeline(config)
        
        # Run preprocessing
        logger.info(f"Starting preprocessing: {args.input} -> {args.output}")
        result_df = pipeline.run_full_pipeline(args.input, args.output)
        
        # Print summary
        print(f"✅ Preprocessing completed successfully!")
        print(f"📊 Processed {len(result_df)} rows")
        print(f"📁 Input: {args.input}")
        print(f"💾 Output: {args.output}")
        print(f"🔧 Config: {args.config}")
        
        # Show basic stats
        if not result_df.empty:
            print(f"\n📈 Data Summary:")
            print(f"   Time range: {result_df['timestamp'].min()} to {result_df['timestamp'].max()}")
            if 'mid_price' in result_df.columns:
                print(f"   Mid price range: {result_df['mid_price'].min():.4f} - {result_df['mid_price'].max():.4f}")
            if 'spread' in result_df.columns:
                print(f"   Average spread: {result_df['spread'].mean():.4f}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()