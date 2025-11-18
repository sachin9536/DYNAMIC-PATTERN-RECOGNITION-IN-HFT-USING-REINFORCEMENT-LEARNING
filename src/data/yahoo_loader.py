"""Yahoo Finance data loader for market data."""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class YahooFinanceLoader:
    """Load market data from Yahoo Finance and convert to our format."""
    
    def __init__(self, symbols: List[str], period: str = "1mo", interval: str = "1m"):
        """
        Initialize Yahoo Finance loader.
        
        Args:
            symbols: List of ticker symbols (e.g., ['AAPL', 'MSFT'])
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        """
        self.symbols = symbols
        self.period = period
        self.interval = interval
        
        logger.info(f"YahooFinanceLoader initialized: {symbols}, period={period}, interval={interval}")
    
    def download_data(self, symbol: str) -> pd.DataFrame:
        """Download data for a single symbol."""
        logger.info(f"Downloading {symbol} data...")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=self.period, interval=self.interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            logger.info(f"Downloaded {len(data)} rows for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to download {symbol}: {e}")
            raise
    
    def convert_to_market_format(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Convert Yahoo Finance data to our market data format.
        
        Yahoo Finance provides: Open, High, Low, Close, Volume
        We need: timestamp, bid_price, ask_price, bid_size, ask_size, trade_price, trade_volume
        """
        logger.info(f"Converting {symbol} data to market format...")
        
        # Reset index to get timestamp as column
        df = data.reset_index()
        
        # Rename timestamp column
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'timestamp'})
        elif 'Date' in df.columns:
            df = df.rename(columns={'Date': 'timestamp'})
        
        # Create synthetic bid/ask from OHLC data
        # This is an approximation - real HFT data would have actual bid/ask
        df['mid_price'] = (df['High'] + df['Low']) / 2
        df['spread'] = (df['High'] - df['Low']) * 0.1  # Assume spread is 10% of range
        
        df['bid_price'] = df['mid_price'] - df['spread'] / 2
        df['ask_price'] = df['mid_price'] + df['spread'] / 2
        
        # Create synthetic sizes (normally distributed around volume)
        np.random.seed(42)  # For reproducibility
        volume_per_tick = df['Volume'] / 100  # Assume 100 ticks per bar
        
        df['bid_size'] = np.maximum(1, np.random.normal(volume_per_tick * 0.4, volume_per_tick * 0.1))
        df['ask_size'] = np.maximum(1, np.random.normal(volume_per_tick * 0.4, volume_per_tick * 0.1))
        
        # Use Close as trade price, Volume as trade volume
        df['trade_price'] = df['Close']
        df['trade_volume'] = df['Volume']
        
        # Select and reorder columns to match our format
        market_columns = [
            'timestamp', 'bid_price', 'ask_price', 'bid_size', 'ask_size', 
            'trade_price', 'trade_volume'
        ]
        
        result = df[market_columns].copy()
        
        # Ensure proper data types
        result['timestamp'] = pd.to_datetime(result['timestamp'])
        for col in ['bid_price', 'ask_price', 'trade_price']:
            result[col] = result[col].astype(float)
        for col in ['bid_size', 'ask_size', 'trade_volume']:
            result[col] = result[col].astype(float)
        
        logger.info(f"Converted to market format: {len(result)} rows")
        return result
    
    def load_multiple_symbols(self) -> Dict[str, pd.DataFrame]:
        """Load data for all symbols."""
        results = {}
        
        for symbol in self.symbols:
            try:
                raw_data = self.download_data(symbol)
                market_data = self.convert_to_market_format(raw_data, symbol)
                results[symbol] = market_data
                
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                continue
        
        return results
    
    def save_to_csv(self, output_dir: str = "data/raw/yahoo") -> Dict[str, str]:
        """Download and save data to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        data_dict = self.load_multiple_symbols()
        saved_files = {}
        
        for symbol, data in data_dict.items():
            filename = f"{symbol}_{self.period}_{self.interval}.csv"
            filepath = output_dir / filename
            
            data.to_csv(filepath, index=False)
            saved_files[symbol] = str(filepath)
            
            logger.info(f"Saved {symbol} data to {filepath}")
        
        return saved_files
    
    def create_combined_dataset(self, min_samples: int = 1000) -> pd.DataFrame:
        """
        Create a combined dataset from multiple symbols.
        Useful for training on diverse market conditions.
        """
        logger.info("Creating combined dataset...")
        
        data_dict = self.load_multiple_symbols()
        combined_data = []
        
        for symbol, data in data_dict.items():
            if len(data) >= min_samples:
                # Add symbol identifier
                data_copy = data.copy()
                data_copy['symbol'] = symbol
                combined_data.append(data_copy)
                logger.info(f"Added {len(data)} samples from {symbol}")
            else:
                logger.warning(f"Skipping {symbol}: only {len(data)} samples (min: {min_samples})")
        
        if not combined_data:
            raise ValueError("No symbols had sufficient data")
        
        # Combine all data
        result = pd.concat(combined_data, ignore_index=True)
        
        # Sort by timestamp
        result = result.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Combined dataset: {len(result)} total samples from {len(combined_data)} symbols")
        return result


def create_sample_yahoo_data():
    """Create sample data using Yahoo Finance for testing."""
    # Popular stocks with good liquidity
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    # Use 1-minute data for better granularity
    loader = YahooFinanceLoader(
        symbols=symbols[:2],  # Start with just 2 symbols
        period="5d",  # Last 5 days
        interval="1m"  # 1-minute intervals
    )
    
    # Save individual files
    saved_files = loader.save_to_csv()
    
    # Create combined dataset
    combined_data = loader.create_combined_dataset(min_samples=100)
    
    # Save combined dataset
    output_path = "data/raw/yahoo_combined.csv"
    combined_data.to_csv(output_path, index=False)
    
    return output_path, saved_files


if __name__ == "__main__":
    try:
        print("Creating Yahoo Finance sample data...")
        
        combined_path, individual_files = create_sample_yahoo_data()
        
        print(f"✅ Created combined dataset: {combined_path}")
        print(f"✅ Individual files: {list(individual_files.values())}")
        
        # Show sample of the data
        df = pd.read_csv(combined_path)
        print(f"\n📊 Combined dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"📅 Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"📈 Symbols: {df['symbol'].unique()}")
        
        print("\nSample data:")
        print(df.head())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()