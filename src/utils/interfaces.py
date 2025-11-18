"""Base interfaces and abstract classes for the system."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np


class ModelInterface(ABC):
    """Abstract interface for all models."""
    
    @abstractmethod
    def predict(self, observation: Union[np.ndarray, List[float]]) -> Dict[str, Any]:
        """Make a prediction on the given observation."""
        pass
    
    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load model from file."""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get model information and metadata."""
        pass


class ExplainerInterface(ABC):
    """Abstract interface for explainability methods."""
    
    @abstractmethod
    def explain(self, model: ModelInterface, observation: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Generate explanation for the given observation."""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get the name of the explanation method."""
        pass


class RuleEngineInterface(ABC):
    """Abstract interface for rule engines."""
    
    @abstractmethod
    def check_rules(self, observation: Dict[str, float]) -> Dict[str, Any]:
        """Check rules against the given observation."""
        pass
    
    @abstractmethod
    def get_rule_summary(self) -> Dict[str, Any]:
        """Get summary of available rules."""
        pass


class DataSourceInterface(ABC):
    """Abstract interface for data sources."""
    
    @abstractmethod
    def get_data(self, limit: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """Get data from the source."""
        pass
    
    @abstractmethod
    def get_real_time_data(self) -> Dict[str, Any]:
        """Get real-time data point."""
        pass


class CacheInterface(ABC):
    """Abstract interface for caching systems."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass


class MetricsInterface(ABC):
    """Abstract interface for metrics collection."""
    
    @abstractmethod
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        pass
    
    @abstractmethod
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        pass
    
    @abstractmethod
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a histogram metric."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        pass


class LoggerInterface(ABC):
    """Abstract interface for structured logging."""
    
    @abstractmethod
    def log_decision(self, decision_data: Dict[str, Any]) -> None:
        """Log a decision with structured data."""
        pass
    
    @abstractmethod
    def log_audit(self, audit_data: Dict[str, Any]) -> None:
        """Log audit information."""
        pass
    
    @abstractmethod
    def log_error(self, error_data: Dict[str, Any]) -> None:
        """Log error information."""
        pass


class SimulationInterface(ABC):
    """Abstract interface for simulation engines."""
    
    @abstractmethod
    def start(self, config: Dict[str, Any]) -> None:
        """Start the simulation."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the simulation."""
        pass
    
    @abstractmethod
    def pause(self) -> None:
        """Pause the simulation."""
        pass
    
    @abstractmethod
    def resume(self) -> None:
        """Resume the simulation."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get simulation status."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get simulation metrics."""
        pass


class ComponentInterface(ABC):
    """Base interface for all system components."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the component with configuration."""
        pass
    
    @abstractmethod
    def start(self) -> None:
        """Start the component."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the component."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, str]:
        """Get component status."""
        pass
    
    @abstractmethod
    def get_health(self) -> Dict[str, Any]:
        """Get component health information."""
        pass


# Common data structures
class PredictionResult:
    """Standard prediction result structure."""
    
    def __init__(self, action: int, scores: List[float], confidence: float, 
                 explanation_hash: Optional[str] = None, processing_time_ms: float = 0.0):
        self.action = action
        self.scores = scores
        self.confidence = confidence
        self.explanation_hash = explanation_hash
        self.processing_time_ms = processing_time_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'action': self.action,
            'scores': self.scores,
            'confidence': self.confidence,
            'explanation_hash': self.explanation_hash,
            'processing_time_ms': self.processing_time_ms
        }


class RuleResult:
    """Standard rule check result structure."""
    
    def __init__(self, rule_flags: Dict[str, bool], triggered_rules: List[str], 
                 anomaly_score: float, explanation_text: str = ""):
        self.rule_flags = rule_flags
        self.triggered_rules = triggered_rules
        self.anomaly_score = anomaly_score
        self.explanation_text = explanation_text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rule_flags': self.rule_flags,
            'triggered_rules': self.triggered_rules,
            'anomaly_score': self.anomaly_score,
            'explanation_text': self.explanation_text
        }


class SystemStatus:
    """Standard system status structure."""
    
    def __init__(self, status: str, timestamp: str, version: str, 
                 uptime_seconds: float, components: Dict[str, str]):
        self.status = status
        self.timestamp = timestamp
        self.version = version
        self.uptime_seconds = uptime_seconds
        self.components = components
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'status': self.status,
            'timestamp': self.timestamp,
            'version': self.version,
            'uptime_seconds': self.uptime_seconds,
            'components': self.components
        }


# Exception classes
class ModelError(Exception):
    """Exception raised for model-related errors."""
    pass


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass


class ResourceError(Exception):
    """Exception raised for resource-related errors."""
    pass


class SimulationError(Exception):
    """Exception raised for simulation-related errors."""
    pass


if __name__ == "__main__":
    # Test interfaces and data structures
    try:
        print("Testing interfaces and data structures...")
        
        # Test PredictionResult
        pred_result = PredictionResult(
            action=1,
            scores=[0.1, 0.8, 0.1],
            confidence=0.85,
            explanation_hash="abc123",
            processing_time_ms=15.5
        )
        print(f"Prediction result: {pred_result.to_dict()}")
        
        # Test RuleResult
        rule_result = RuleResult(
            rule_flags={'high_volatility': True, 'volume_spike': False},
            triggered_rules=['high_volatility'],
            anomaly_score=0.75,
            explanation_text="High volatility detected"
        )
        print(f"Rule result: {rule_result.to_dict()}")
        
        # Test SystemStatus
        system_status = SystemStatus(
            status="healthy",
            timestamp="2024-01-15T10:30:00Z",
            version="1.0.0",
            uptime_seconds=3600.0,
            components={'api': 'running', 'dashboard': 'running'}
        )
        print(f"System status: {system_status.to_dict()}")
        
        print("✅ Interfaces and data structures test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()