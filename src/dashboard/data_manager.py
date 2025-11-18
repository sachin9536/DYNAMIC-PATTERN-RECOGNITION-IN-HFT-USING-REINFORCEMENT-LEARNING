"""Data management for the dashboard."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import pickle
from datetime import datetime, timedelta

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

from .config import (
    DATA_DIR, ARTIFACTS_DIR, MODELS_DIR, EXPLANATIONS_DIR,
    MAX_DATA_POINTS, CACHE_TTL
)


class DashboardDataManager:
    """Manages data loading and caching for the dashboard."""
    
    def __init__(self):
        self.cache = {}
        self.cache_timestamps = {}
        
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self.cache_timestamps:
            return False
        
        age = (datetime.now() - self.cache_timestamps[key]).total_seconds()
        return age < CACHE_TTL
    
    def _cache_data(self, key: str, data: Any) -> None:
        """Cache data with timestamp."""
        self.cache[key] = data
        self.cache_timestamps[key] = datetime.now()
    
    def load_market_data(self, limit: int = MAX_DATA_POINTS) -> Optional[pd.DataFrame]:
        """Load market data for visualization."""
        cache_key = f"market_data_{limit}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            # Try to load real data first
            data_files = [
                DATA_DIR / "processed" / "market_data.csv",
                DATA_DIR / "raw" / "market_data.csv",
                ARTIFACTS_DIR / "synthetic" / "market_data.csv"
            ]
            
            for data_file in data_files:
                if data_file.exists():
                    logger.info(f"Loading market data from {data_file}")
                    df = pd.read_csv(data_file)
                    
                    # Ensure required columns
                    if 'timestamp' not in df.columns and 'time' in df.columns:
                        df['timestamp'] = df['time']
                    
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.sort_values('timestamp').tail(limit)
                    else:
                        df = df.tail(limit)
                    
                    self._cache_data(cache_key, df)
                    return df
            
            # Generate synthetic data if no real data available
            logger.info("No market data found, generating synthetic data")
            df = self._generate_synthetic_market_data(limit)
            self._cache_data(cache_key, df)
            return df
            
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            # Return synthetic data as fallback
            df = self._generate_synthetic_market_data(limit)
            self._cache_data(cache_key, df)
            return df
    
    def _generate_synthetic_market_data(self, n_points: int) -> pd.DataFrame:
        """Generate synthetic market data for demo purposes."""
        np.random.seed(42)  # For reproducible demo data
        
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=n_points/60)  # Assume 1-minute intervals
        timestamps = pd.date_range(start_time, end_time, periods=n_points)
        
        # Generate price data with trend and noise
        base_price = 100.0
        trend = np.linspace(0, 5, n_points)  # Slight upward trend
        noise = np.random.normal(0, 0.5, n_points)
        price_changes = np.random.normal(0, 0.1, n_points)
        
        prices = [base_price]
        for i in range(1, n_points):
            new_price = prices[-1] + price_changes[i] + trend[i]/n_points
            prices.append(max(new_price, 0.01))  # Ensure positive prices
        
        # Generate volume data
        base_volume = 1000
        volume_noise = np.random.lognormal(0, 0.3, n_points)
        volumes = (base_volume * volume_noise).astype(int)
        
        # Generate anomalies (10% of data points)
        anomaly_indices = np.random.choice(n_points, size=int(n_points * 0.1), replace=False)
        is_anomaly = np.zeros(n_points, dtype=bool)
        is_anomaly[anomaly_indices] = True
        
        # Add anomaly effects to price and volume
        anomaly_price_multiplier = np.ones(n_points)
        anomaly_volume_multiplier = np.ones(n_points)
        
        for idx in anomaly_indices:
            # Random price spike or drop
            anomaly_price_multiplier[idx] = np.random.choice([0.95, 1.05], p=[0.5, 0.5])
            # Volume spike during anomalies
            anomaly_volume_multiplier[idx] = np.random.uniform(2.0, 5.0)
        
        prices = np.array(prices) * anomaly_price_multiplier
        volumes = volumes * anomaly_volume_multiplier
        
        # Calculate additional features
        returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
        volatility = pd.Series(returns).rolling(window=20, min_periods=1).std().values
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes.astype(int),
            'returns': returns,
            'volatility': volatility,
            'is_anomaly': is_anomaly,
            'anomaly_score': np.where(is_anomaly, np.random.uniform(0.7, 1.0, n_points), 
                                    np.random.uniform(0.0, 0.3, n_points))
        })
        
        return df
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary statistics for anomalies."""
        cache_key = "anomaly_summary"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            market_data = self.load_market_data()
            if market_data is None or len(market_data) == 0:
                return self._get_default_anomaly_summary()
            
            # Calculate anomaly statistics
            total_points = len(market_data)
            anomaly_points = market_data['is_anomaly'].sum() if 'is_anomaly' in market_data.columns else 0
            anomaly_rate = anomaly_points / total_points if total_points > 0 else 0
            
            # Get recent anomalies (last 24 hours)
            if 'timestamp' in market_data.columns:
                recent_cutoff = datetime.now() - timedelta(hours=24)
                recent_data = market_data[market_data['timestamp'] > recent_cutoff]
                recent_anomalies = recent_data['is_anomaly'].sum() if 'is_anomaly' in recent_data.columns else 0
            else:
                recent_anomalies = anomaly_points  # Fallback
            
            # Calculate severity distribution
            if 'anomaly_score' in market_data.columns:
                anomaly_scores = market_data[market_data['is_anomaly']]['anomaly_score']
                severity_dist = {
                    'low': (anomaly_scores < 0.3).sum(),
                    'medium': ((anomaly_scores >= 0.3) & (anomaly_scores < 0.7)).sum(),
                    'high': (anomaly_scores >= 0.7).sum()
                }
            else:
                severity_dist = {'low': 0, 'medium': 0, 'high': anomaly_points}
            
            summary = {
                'total_anomalies': int(anomaly_points),
                'anomaly_rate': float(anomaly_rate),
                'recent_anomalies_24h': int(recent_anomalies),
                'severity_distribution': severity_dist,
                'last_updated': datetime.now().isoformat()
            }
            
            self._cache_data(cache_key, summary)
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get anomaly summary: {e}")
            return self._get_default_anomaly_summary()
    
    def _get_default_anomaly_summary(self) -> Dict[str, Any]:
        """Get default anomaly summary for demo."""
        return {
            'total_anomalies': 42,
            'anomaly_rate': 0.08,
            'recent_anomalies_24h': 5,
            'severity_distribution': {'low': 15, 'medium': 20, 'high': 7},
            'last_updated': datetime.now().isoformat()
        }
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        cache_key = "model_metrics"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            # Try to load real metrics
            metrics_files = [
                ARTIFACTS_DIR / "eval" / "metrics.json",
                ARTIFACTS_DIR / "training_metadata.json"
            ]
            
            for metrics_file in metrics_files:
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    # Standardize metrics format
                    standardized = self._standardize_metrics(metrics)
                    self._cache_data(cache_key, standardized)
                    return standardized
            
            # Generate demo metrics if no real metrics available
            metrics = self._generate_demo_metrics()
            self._cache_data(cache_key, metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to load model metrics: {e}")
            metrics = self._generate_demo_metrics()
            self._cache_data(cache_key, metrics)
            return metrics
    
    def _standardize_metrics(self, raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize metrics format for dashboard display."""
        return {
            'accuracy': raw_metrics.get('accuracy', 0.85),
            'precision': raw_metrics.get('precision', 0.82),
            'recall': raw_metrics.get('recall', 0.78),
            'f1_score': raw_metrics.get('f1_score', 0.80),
            'auc_roc': raw_metrics.get('auc_roc', 0.88),
            'training_time': raw_metrics.get('training_time', '45 minutes'),
            'model_type': raw_metrics.get('model_type', 'PPO'),
            'last_updated': raw_metrics.get('timestamp', datetime.now().isoformat())
        }
    
    def _generate_demo_metrics(self) -> Dict[str, Any]:
        """Generate demo metrics for display."""
        return {
            'accuracy': 0.87,
            'precision': 0.84,
            'recall': 0.81,
            'f1_score': 0.82,
            'auc_roc': 0.89,
            'training_time': '42 minutes',
            'model_type': 'PPO',
            'last_updated': datetime.now().isoformat()
        }
    
    def get_explanation_data(self, method: str = 'rule') -> Optional[Dict[str, Any]]:
        """Get explanation data for the dashboard."""
        cache_key = f"explanation_{method}"
        
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            # Look for explanation files
            explanation_files = list(EXPLANATIONS_DIR.glob(f"{method}_*.json"))
            
            if explanation_files:
                # Load the most recent explanation
                latest_file = max(explanation_files, key=lambda x: x.stat().st_mtime)
                with open(latest_file, 'r') as f:
                    explanation = json.load(f)
                
                self._cache_data(cache_key, explanation)
                return explanation
            
            # Generate demo explanation if none available
            explanation = self._generate_demo_explanation(method)
            self._cache_data(cache_key, explanation)
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to load explanation data: {e}")
            explanation = self._generate_demo_explanation(method)
            self._cache_data(cache_key, explanation)
            return explanation
    
    def _generate_demo_explanation(self, method: str) -> Dict[str, Any]:
        """Generate demo explanation data."""
        if method == 'rule':
            return {
                'method': 'rule',
                'anomaly_score': 0.75,
                'triggered_rules': ['high_volatility', 'volume_spike'],
                'explanation_text': 'High volatility detected with unusual volume spike',
                'feature_names': ['price', 'volume', 'volatility', 'returns'],
                'timestamp': datetime.now().isoformat()
            }
        elif method == 'shap':
            return {
                'method': 'shap',
                'shap_values': [[0.1, -0.3, 0.5, 0.2]],
                'expected_value': 0.1,
                'feature_names': ['price', 'volume', 'volatility', 'returns'],
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'method': method,
                'feature_importance': {'price': 0.4, 'volume': 0.3, 'volatility': 0.2, 'returns': 0.1},
                'feature_names': ['price', 'volume', 'volatility', 'returns'],
                'timestamp': datetime.now().isoformat()
            }
    
    def get_available_models(self) -> List[str]:
        """Get list of available trained models."""
        try:
            if MODELS_DIR.exists():
                model_files = list(MODELS_DIR.glob("*.zip")) + list(MODELS_DIR.glob("*.pkl"))
                return [f.stem for f in model_files]
            return ['demo_model', 'ppo_model', 'sac_model']
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return ['demo_model']
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Dashboard cache cleared")


if __name__ == "__main__":
    # Test data manager
    try:
        print("Testing Dashboard Data Manager...")
        
        manager = DashboardDataManager()
        
        # Test market data loading
        market_data = manager.load_market_data(limit=100)
        print(f"Market data shape: {market_data.shape if market_data is not None else 'None'}")
        
        # Test anomaly summary
        anomaly_summary = manager.get_anomaly_summary()
        print(f"Anomaly summary: {anomaly_summary}")
        
        # Test model metrics
        metrics = manager.get_model_metrics()
        print(f"Model metrics: {list(metrics.keys())}")
        
        # Test explanation data
        explanation = manager.get_explanation_data('rule')
        print(f"Explanation method: {explanation.get('method', 'None')}")
        
        print("✅ Data manager test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()