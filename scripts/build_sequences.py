#!/usr/bin/env python3
"""Command-line interface for building sequences from preprocessed data."""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engineering import compute_base_features, normalize_features
from src.data.sequence_builder import build_sequences, build_targets, save_sequences
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Build sequences from preprocessed market data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input preprocessed CSV file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output sequences file (.npz)"
    )
    
    parser.add_argument(
        "--seq_len",
        type=int,
        default=100,
        help="Length of each sequence"
    )
    
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size between sequences (for sliding mode)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="sliding",
        choices=["sliding", "chunk"],
        help="Sequence building mode"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--target_col",
        type=str,
        default="mid_price",
        help="Column to use for target generation"
    )
    
    parser.add_argument(
        "--target_horizon",
        type=int,
        default=1,
        help="Prediction horizon for targets"
    )
    
    parser.add_argument(
        "--skip_features",
        action="store_true",
        help="Skip feature engineering (assume input already has features)"
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
        
        # Load input data
        logger.info(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Feature engineering (if not skipped)
        if not args.skip_features:
            logger.info("Computing base features...")
            df = compute_base_features(df)
            
            # Get normalization columns - use base features, not preprocessing columns
            normalize_cols = ['mid_price', 'spread', 'order_imbalance', 'trade_intensity', 'rolling_vol_10']
            
            logger.info(f"Normalizing features: {normalize_cols}")
            df, scalers = normalize_features(df, normalize_cols)
            
            logger.info(f"Feature engineering complete. Shape: {df.shape}")
        
        # Get feature columns from config or auto-detect
        feature_cols = config.get('features', {}).get('feature_cols')
        if feature_cols is None:
            # Auto-detect normalized columns
            feature_cols = [col for col in df.columns if col.endswith('_z')]
            logger.info(f"Auto-detected feature columns: {feature_cols}")
        
        # Filter feature columns to only include those that exist
        existing_feature_cols = [col for col in feature_cols if col in df.columns]
        if not existing_feature_cols:
            raise ValueError("No feature columns found. Check your data or configuration.")
        
        if len(existing_feature_cols) != len(feature_cols):
            logger.warning(f"Some feature columns missing. Using: {existing_feature_cols}")
            feature_cols = existing_feature_cols
        
        # Build sequences
        logger.info(f"Building sequences: seq_len={args.seq_len}, step={args.step}, mode={args.mode}")
        sequences = build_sequences(
            df,
            seq_len=args.seq_len,
            step=args.step,
            mode=args.mode,
            feature_cols=feature_cols
        )
        
        # Build targets
        logger.info(f"Building targets: horizon={args.target_horizon}, target_col={args.target_col}")
        targets = build_targets(
            df,
            horizon=args.target_horizon,
            target_col=args.target_col,
            target_type='return'
        )
        
        # Align targets with sequences
        targets_aligned = targets[:len(sequences)]
        
        # Prepare metadata
        metadata = {
            'feature_cols': feature_cols,
            'seq_len': args.seq_len,
            'step': args.step,
            'mode': args.mode,
            'target_col': args.target_col,
            'target_horizon': args.target_horizon,
            'n_original_samples': len(df)
        }
        
        # Save sequences
        logger.info(f"Saving sequences to {args.output}")
        save_sequences(sequences, targets_aligned, args.output, metadata)
        
        # Print summary
        print("Sequence building completed successfully!")
        print(f"Built {len(sequences)} sequences")
        print(f"Sequence shape: {sequences.shape}")
        print(f"Targets shape: {targets_aligned.shape}")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        
        # Show feature statistics
        print(f"\nFeature Statistics:")
        print(f"   Features used: {len(feature_cols)}")
        print(f"   Feature names: {feature_cols}")
        
        if len(sequences) > 0:
            print(f"   Sequence value range: [{sequences.min():.4f}, {sequences.max():.4f}]")
            print(f"   Target value range: [{targets_aligned.min():.4f}, {targets_aligned.max():.4f}]")
        
    except Exception as e:
        logger.error(f"Sequence building failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()