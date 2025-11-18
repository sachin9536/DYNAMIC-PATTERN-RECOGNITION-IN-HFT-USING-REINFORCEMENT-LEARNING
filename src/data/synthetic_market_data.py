"""Generate synthetic market data that mimics real HFT patterns."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
from pathlib import Path

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class SyntheticMarketDataGenerator:
    """Generate realistic synthetic market data for testing and development."""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed."""
        self.seed = seed
        np.random.seed(seed)
        logger.info(f"SyntheticMarketDataGenerator initialized with seed {seed}")
    
    def generate_price_series(
        self, 
        n_samples: int, 
        initial_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.0001,
        dt: float = 0.001  # Time step in hours
    ) -> np.ndarray:
        """
        Generate a realistic price series using geometric Brownian motion.
        
        Args:
            n_samples: Number of price points
            initial_price: Starting price
            volatility: Price volatility (standard deviation)
            trend: Drift term (positive = upward trend)
            dt: Time step
        
        Returns:
            Array of prices
        """
        # Generate random shocks
        shocks = np.random.normal(0, 1, n_samples)
        
        # Geometric Brownian Motion
        log_returns = (trend - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks
        
        # Convert to price series
        prices = np.zeros(n_samples)
        prices[0] = initial_price
        
        for i in range(1, n_samples):
            prices[i] = prices[i-1] * np.exp(log_returns[i])
        
        return prices
    
    def add_microstructure_noise(
        self, 
        prices: np.ndarray, 
        bid_ask_spread: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add realistic bid-ask spread and microstructure noise.
        
        Args:
            prices: Mid prices
            bid_ask_spread: Average bid-ask spread as fraction of price
        
        Returns:
            Tuple of (bid_prices, ask_prices)
        """
        # Variable spread based on volatility
        spread_noise = np.random.normal(1.0, 0.2, len(prices))
        spreads = bid_ask_spread * prices * np.abs(spread_noise)
        
        # Ensure minimum spread
        spreads = np.maximum(spreads, 0.001)
        
        bid_prices = prices - spreads / 2
        ask_prices = prices + spreads / 2
        
        return bid_prices, ask_prices
    
    def generate_order_sizes(
        self, 
        n_samples: int, 
        base_size: float = 100.0,
        size_volatility: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate realistic order sizes for bid and ask.
        
        Args:
            n_samples: Number of samples
            base_size: Base order size
            size_volatility: Volatility of order sizes
        
        Returns:
            Tuple of (bid_sizes, ask_sizes)
        """
        # Log-normal distribution for order sizes
        bid_sizes = np.random.lognormal(
            mean=np.log(base_size), 
            sigma=size_volatility, 
            size=n_samples
        )
        
        ask_sizes = np.random.lognormal(
            mean=np.log(base_size), 
            sigma=size_volatility, 
            size=n_samples
        )
        
        # Round to reasonable precision
        bid_sizes = np.round(bid_sizes, 1)
        ask_sizes = np.round(ask_sizes, 1)
        
        return bid_sizes, ask_sizes
    
    def generate_trade_data(
        self, 
        mid_prices: np.ndarray, 
        bid_prices: np.ndarray, 
        ask_prices: np.ndarray,
        trade_intensity: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trade prices and volumes.
        
        Args:
            mid_prices: Mid prices
            bid_prices: Bid prices
            ask_prices: Ask prices
            trade_intensity: Probability of trade per tick
        
        Returns:
            Tuple of (trade_prices, trade_volumes)
        """
        n_samples = len(mid_prices)
        trade_prices = np.zeros(n_samples)
        trade_volumes = np.zeros(n_samples)
        
        for i in range(n_samples):
            if np.random.random() < trade_intensity:
                # Random trade between bid and ask
                trade_prices[i] = np.random.uniform(bid_prices[i], ask_prices[i])
                trade_volumes[i] = np.random.exponential(50)  # Exponential distribution
            else:
                # No trade - use mid price and zero volume
                trade_prices[i] = mid_prices[i]
                trade_volumes[i] = 0.0
        
        return trade_prices, trade_volumes
    
    def add_anomalies(
        self, 
        data: pd.DataFrame, 
        anomaly_probability: float = 0.05,
        anomaly_strength: float = 3.0
    ) -> pd.DataFrame:
        """
        Add realistic market anomalies to the data.
        
        Args:
            data: Market data DataFrame
            anomaly_probability: Probability of anomaly per sample
            anomaly_strength: Strength of anomalies (in standard deviations)
        
        Returns:
            DataFrame with anomalies added
        """
        result = data.copy()
        n_samples = len(result)
        
        # Add price anomalies (sudden jumps)
        price_anomalies = np.random.random(n_samples) < anomaly_probability
        if price_anomalies.any():
            price_shocks = np.random.normal(0, anomaly_strength, n_samples) * price_anomalies
            price_multiplier = np.exp(price_shocks * 0.01)  # Convert to multiplicative shocks
            
            result.loc[price_anomalies, 'bid_price'] *= price_multiplier[price_anomalies]
            result.loc[price_anomalies, 'ask_price'] *= price_multiplier[price_anomalies]
            result.loc[price_anomalies, 'trade_price'] *= price_multiplier[price_anomalies]
        
        # Add volume anomalies (unusual order sizes)
        volume_anomalies = np.random.random(n_samples) < anomaly_probability * 0.5
        if volume_anomalies.any():
            volume_multiplier = np.random.lognormal(2, 1, n_samples)  # Large volume spikes
            result.loc[volume_anomalies, 'bid_size'] *= volume_multiplier[volume_anomalies]
            result.loc[volume_anomalies, 'ask_size'] *= volume_multiplier[volume_anomalies]
            result.loc[volume_anomalies, 'trade_volume'] *= volume_multiplier[volume_anomalies]
        
        # Add spread anomalies (unusual spreads)
        spread_anomalies = np.random.random(n_samples) < anomaly_probability * 0.3
        if spread_anomalies.any():
            spread_multiplier = np.random.lognormal(1, 0.5, n_samples)
            current_spread = result['ask_price'] - result['bid_price']
            new_spread = current_spread * spread_multiplier
            
            result.loc[spread_anomalies, 'bid_price'] = (
                result.loc[spread_anomalies, 'trade_price'] - new_spread[spread_anomalies] / 2
            )
            result.loc[spread_anomalies, 'ask_price'] = (
                result.loc[spread_anomalies, 'trade_price'] + new_spread[spread_anomalies] / 2
            )
        
        logger.info(f"Added anomalies: {price_anomalies.sum()} price, {volume_anomalies.sum()} volume, {spread_anomalies.sum()} spread")
        
        return result
    
    def generate_market_data(
        self,
        n_samples: int = 10000,
        start_time: Optional[datetime] = None,
        time_interval_ms: int = 100,
        initial_price: float = 100.0,
        volatility: float = 0.02,
        add_anomalies: bool = True
    ) -> pd.DataFrame:
        """
        Generate complete synthetic market data.
        
        Args:
            n_samples: Number of data points
            start_time: Start timestamp (default: now - n_samples * interval)
            time_interval_ms: Time interval between samples in milliseconds
            initial_price: Starting price
            volatility: Price volatility
            add_anomalies: Whether to add market anomalies
        
        Returns:
            DataFrame with market data
        """
        logger.info(f"Generating {n_samples} samples of synthetic market data...")
        
        # Generate timestamps
        if start_time is None:
            start_time = datetime.now() - timedelta(milliseconds=n_samples * time_interval_ms)
        
        timestamps = [
            start_time + timedelta(milliseconds=i * time_interval_ms) 
            for i in range(n_samples)
        ]
        
        # Generate price series
        mid_prices = self.generate_price_series(
            n_samples, initial_price, volatility
        )
        
        # Generate bid/ask prices
        bid_prices, ask_prices = self.add_microstructure_noise(mid_prices)
        
        # Generate order sizes
        bid_sizes, ask_sizes = self.generate_order_sizes(n_samples)
        
        # Generate trade data
        trade_prices, trade_volumes = self.generate_trade_data(
            mid_prices, bid_prices, ask_prices
        )
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'bid_price': bid_prices,
            'ask_price': ask_prices,
            'bid_size': bid_sizes,
            'ask_size': ask_sizes,
            'trade_price': trade_prices,
            'trade_volume': trade_volumes
        })
        
        # Add anomalies if requested
        if add_anomalies:
            data = self.add_anomalies(data)
        
        logger.info(f"Generated synthetic market data: {len(data)} samples")
        logger.info(f"Price range: {data['trade_price'].min():.2f} - {data['trade_price'].max():.2f}")
        logger.info(f"Time range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        
        return data
    
    def save_to_csv(self, data: pd.DataFrame, filepath: str) -> str:
        """Save market data to CSV file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_csv(filepath, index=False)
        logger.info(f"Saved {len(data)} samples to {filepath}")
        
        return str(filepath)


def create_realistic_market_data(
    n_samples: int = 5000,
    output_path: str = "data/raw/synthetic_market_data.csv"
) -> str:
    """Create realistic synthetic market data for testing."""
    generator = SyntheticMarketDataGenerator(seed=42)
    
    # Generate data with realistic parameters
    data = generator.generate_market_data(
        n_samples=n_samples,
        time_interval_ms=50,  # 50ms intervals (20 Hz)
        initial_price=100.0,
        volatility=0.015,  # 1.5% volatility
        add_anomalies=True
    )
    
    # Save to file
    filepath = generator.save_to_csv(data, output_path)
    
    return filepath


if __name__ == "__main__":
    try:
        print("🔄 Generating synthetic market data...")
        
        filepath = create_realistic_market_data(n_samples=10000)
        
        # Load and show sample
        df = pd.read_csv(filepath)
        
        print(f"✅ Created synthetic market data: {filepath}")
        print(f"📊 {len(df)} samples, {len(df.columns)} columns")
        print(f"📅 Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"💰 Price range: ${df['trade_price'].min():.2f} - ${df['trade_price'].max():.2f}")
        
        print("\n📈 Sample data:")
        print(df.head(10))
        
        print(f"\n🚀 Ready to process with:")
        print(f"python scripts/run_preprocessing.py --input {filepath} --output data/processed/synthetic_processed.csv")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()