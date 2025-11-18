#!/usr/bin/env python3
"""Script to download real market data from Yahoo Finance."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.yahoo_loader import YahooFinanceLoader, create_sample_yahoo_data


def main():
    """Main function to download Yahoo Finance data."""
    parser = argparse.ArgumentParser(description="Download market data from Yahoo Finance")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "GOOGL"],
                       help="Stock symbols to download")
    parser.add_argument("--period", default="1mo", 
                       choices=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                       help="Data period")
    parser.add_argument("--interval", default="1m",
                       choices=["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
                       help="Data interval")
    parser.add_argument("--output", default="data/raw/yahoo",
                       help="Output directory")
    parser.add_argument("--combined", action="store_true",
                       help="Create combined dataset from all symbols")
    
    args = parser.parse_args()
    
    try:
        print(f"📊 Downloading data for: {args.symbols}")
        print(f"📅 Period: {args.period}, Interval: {args.interval}")
        
        # Create loader
        loader = YahooFinanceLoader(
            symbols=args.symbols,
            period=args.period,
            interval=args.interval
        )
        
        # Save individual files
        saved_files = loader.save_to_csv(args.output)
        
        print(f"\n✅ Downloaded {len(saved_files)} symbols:")
        for symbol, filepath in saved_files.items():
            print(f"   {symbol}: {filepath}")
        
        # Create combined dataset if requested
        if args.combined:
            combined_data = loader.create_combined_dataset(min_samples=100)
            combined_path = Path(args.output) / f"combined_{args.period}_{args.interval}.csv"
            combined_data.to_csv(combined_path, index=False)
            
            print(f"\n✅ Combined dataset: {combined_path}")
            print(f"   📊 {len(combined_data)} total samples")
            print(f"   📈 Symbols: {combined_data['symbol'].unique()}")
            print(f"   📅 Time range: {combined_data['timestamp'].min()} to {combined_data['timestamp'].max()}")
        
        print(f"\n🚀 Ready to process with:")
        print(f"   python scripts/run_preprocessing.py --input {list(saved_files.values())[0]} --output data/processed/yahoo_processed.csv")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())