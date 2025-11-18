"""Real-time data streaming for the dashboard."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generator
import time
import threading
from collections import deque
import queue

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

from .config import STREAM_BUFFER_SIZE, STREAM_UPDATE_FREQUENCY


class RealTimeDataStream:
    """Manages real-time data streaming for the dashboard."""
    
    def __init__(self, buffer_size: int = STREAM_BUFFER_SIZE):
        self.buffer_size = buffer_size
        self.data_buffer = deque(maxlen=buffer_size)
        self.is_streaming = False
        self.stream_thread = None
        self.data_queue = queue.Queue()
        self.last_price = 100.0
        self.last_timestamp = datetime.now()
        
    def start_stream(self) -> None:
        """Start the real-time data stream."""
        if self.is_streaming:
            logger.warning("Stream is already running")
            return
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()
        logger.info("Real-time data stream started")
    
    def stop_stream(self) -> None:
        """Stop the real-time data stream."""
        self.is_streaming = False
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=1.0)
        logger.info("Real-time data stream stopped")
    
    def _stream_worker(self) -> None:
        """Worker thread for generating streaming data."""
        while self.is_streaming:
            try:
                # Generate new data point
                new_data = self._generate_next_data_point()
                
                # Add to buffer and queue
                self.data_buffer.append(new_data)
                self.data_queue.put(new_data)
                
                # Sleep until next update
                time.sleep(STREAM_UPDATE_FREQUENCY)
                
            except Exception as e:
                logger.error(f"Error in stream worker: {e}")
                time.sleep(1.0)  # Brief pause before retrying
    
    def _generate_next_data_point(self) -> Dict[str, Any]:
        """Generate the next data point in the stream."""
        current_time = datetime.now()
        
        # Generate price with random walk
        price_change = np.random.normal(0, 0.1)  # Small random changes
        new_price = max(self.last_price + price_change, 0.01)  # Ensure positive
        
        # Generate volume
        base_volume = 1000
        volume_multiplier = np.random.lognormal(0, 0.2)
        volume = int(base_volume * volume_multiplier)
        
        # Calculate returns
        returns = np.log(new_price / self.last_price) if self.last_price > 0 else 0
        
        # Generate anomaly (5% chance)
        is_anomaly = np.random.random() < 0.05
        if is_anomaly:
            # Add anomaly effects
            new_price *= np.random.choice([0.95, 1.05])  # Price spike/drop
            volume *= np.random.uniform(2.0, 4.0)  # Volume spike
        
        # Calculate anomaly score
        anomaly_score = np.random.uniform(0.7, 1.0) if is_anomaly else np.random.uniform(0.0, 0.3)
        
        # Calculate volatility (simplified)
        if len(self.data_buffer) > 0:
            recent_returns = [point['returns'] for point in list(self.data_buffer)[-20:]]
            volatility = np.std(recent_returns) if recent_returns else 0.1
        else:
            volatility = 0.1
        
        data_point = {
            'timestamp': current_time,
            'price': new_price,
            'volume': int(volume),
            'returns': returns,
            'volatility': volatility,
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score
        }
        
        # Update state
        self.last_price = new_price
        self.last_timestamp = current_time
        
        return data_point
    
    def get_latest_data(self, n_points: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the latest n data points from the buffer."""
        if n_points is None:
            return list(self.data_buffer)
        else:
            return list(self.data_buffer)[-n_points:]
    
    def get_new_data(self) -> List[Dict[str, Any]]:
        """Get all new data points since last call."""
        new_data = []
        while not self.data_queue.empty():
            try:
                new_data.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return new_data
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get statistics about the data stream."""
        if not self.data_buffer:
            return {
                'buffer_size': 0,
                'is_streaming': self.is_streaming,
                'last_update': None,
                'anomaly_rate': 0.0
            }
        
        data_list = list(self.data_buffer)
        anomaly_count = sum(1 for point in data_list if point['is_anomaly'])
        
        return {
            'buffer_size': len(data_list),
            'is_streaming': self.is_streaming,
            'last_update': self.last_timestamp.isoformat(),
            'anomaly_rate': anomaly_count / len(data_list) if data_list else 0.0,
            'current_price': self.last_price,
            'price_range': {
                'min': min(point['price'] for point in data_list),
                'max': max(point['price'] for point in data_list)
            } if data_list else {'min': 0, 'max': 0}
        }
    
    def get_stream_config(self) -> Dict[str, Any]:
        """Get stream configuration."""
        return {
            'buffer_size': self.buffer_size,
            'update_frequency': STREAM_UPDATE_FREQUENCY,
            'is_streaming': self.is_streaming,
            'stream_thread_alive': self.stream_thread.is_alive() if self.stream_thread else False
        }
    
    def generate_sample_data(self, n_points: int = 100) -> pd.DataFrame:
        """Generate sample data for testing/demo purposes."""
        logger.info(f"Generating {n_points} sample data points")
        
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=n_points)
        timestamps = pd.date_range(start_time, end_time, periods=n_points)
        
        # Initialize with current state or defaults
        current_price = self.last_price if hasattr(self, 'last_price') else 100.0
        
        data_points = []
        for i, timestamp in enumerate(timestamps):
            # Generate price with random walk
            price_change = np.random.normal(0, 0.1)
            current_price = max(current_price + price_change, 0.01)
            
            # Generate other features
            volume = int(np.random.lognormal(np.log(1000), 0.3))
            returns = price_change / current_price if current_price > 0 else 0
            
            # Generate anomaly (10% chance)
            is_anomaly = np.random.random() < 0.1
            if is_anomaly:
                current_price *= np.random.choice([0.95, 1.05])
                volume *= np.random.uniform(2.0, 4.0)
            
            anomaly_score = np.random.uniform(0.7, 1.0) if is_anomaly else np.random.uniform(0.0, 0.3)
            
            # Calculate volatility (rolling window)
            if i >= 20:
                recent_returns = [dp['returns'] for dp in data_points[-20:]]
                volatility = np.std(recent_returns)
            else:
                volatility = 0.1
            
            data_point = {
                'timestamp': timestamp,
                'price': current_price,
                'volume': int(volume),
                'returns': returns,
                'volatility': volatility,
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score
            }
            
            data_points.append(data_point)
        
        return pd.DataFrame(data_points)
    
    def clear_buffer(self) -> None:
        """Clear the data buffer."""
        self.data_buffer.clear()
        # Clear the queue as well
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Data buffer cleared")


class StreamingDataGenerator:
    """Generator for streaming data points."""
    
    def __init__(self, initial_price: float = 100.0):
        self.current_price = initial_price
        self.current_time = datetime.now()
        
    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        """Generate infinite stream of data points."""
        while True:
            # Update time
            self.current_time += timedelta(seconds=STREAM_UPDATE_FREQUENCY)
            
            # Generate price change
            price_change = np.random.normal(0, 0.1)
            self.current_price = max(self.current_price + price_change, 0.01)
            
            # Generate other features
            volume = int(np.random.lognormal(np.log(1000), 0.3))
            returns = price_change / self.current_price if self.current_price > 0 else 0
            
            # Generate anomaly
            is_anomaly = np.random.random() < 0.05
            if is_anomaly:
                self.current_price *= np.random.choice([0.95, 1.05])
                volume *= np.random.uniform(2.0, 4.0)
            
            anomaly_score = np.random.uniform(0.7, 1.0) if is_anomaly else np.random.uniform(0.0, 0.3)
            
            yield {
                'timestamp': self.current_time,
                'price': self.current_price,
                'volume': volume,
                'returns': returns,
                'volatility': abs(returns) * 10,  # Simplified volatility
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score
            }


if __name__ == "__main__":
    # Test real-time data stream
    try:
        print("Testing Real-time Data Stream...")
        
        stream = RealTimeDataStream(buffer_size=50)
        
        # Test configuration
        config = stream.get_stream_config()
        print(f"Stream config: {config}")
        
        # Test sample data generation
        sample_data = stream.generate_sample_data(n_points=20)
        print(f"Generated sample data shape: {sample_data.shape}")
        print(f"Sample data columns: {list(sample_data.columns)}")
        
        # Test streaming (brief test)
        print("Starting stream for 3 seconds...")
        stream.start_stream()
        time.sleep(3)
        
        # Get data
        latest_data = stream.get_latest_data(n_points=5)
        print(f"Latest data points: {len(latest_data)}")
        
        # Get stats
        stats = stream.get_stream_stats()
        print(f"Stream stats: {stats}")
        
        # Stop stream
        stream.stop_stream()
        
        print("✅ Real-time stream test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()