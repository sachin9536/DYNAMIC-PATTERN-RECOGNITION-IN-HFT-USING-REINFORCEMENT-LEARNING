"""Monitoring utilities with Prometheus metrics and structured logging."""

import os
import json
import time
import functools
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import threading

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not available. Install with: pip install prometheus-client")

try:
    from src.utils.logger import get_logger
    from src.utils.config_manager import get_monitoring_config
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Prometheus metrics collector."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Set up Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, metrics disabled")
            return
        
        # Counters
        self.predictions_total = Counter(
            'predictions_total',
            'Total number of predictions made',
            ['model_id', 'status'],
            registry=self.registry
        )
        
        self.prediction_errors_total = Counter(
            'prediction_errors_total',
            'Total number of prediction errors',
            ['model_id', 'error_type'],
            registry=self.registry
        )
        
        self.rule_checks_total = Counter(
            'rule_checks_total',
            'Total number of rule checks performed',
            ['status'],
            registry=self.registry
        )
        
        self.api_requests_total = Counter(
            'api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        # Histograms
        self.prediction_latency_seconds = Histogram(
            'prediction_latency_seconds',
            'Prediction latency in seconds',
            ['model_id'],
            registry=self.registry
        )
        
        self.model_load_time_seconds = Histogram(
            'model_load_time_seconds',
            'Model loading time in seconds',
            ['model_id'],
            registry=self.registry
        )
        
        self.api_request_duration_seconds = Histogram(
            'api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Gauges
        self.active_models = Gauge(
            'active_models',
            'Number of currently loaded models',
            registry=self.registry
        )
        
        self.memory_usage_bytes = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage_percent = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.disk_usage_bytes = Gauge(
            'disk_usage_bytes',
            'Disk usage in bytes',
            registry=self.registry
        )
        
        logger.info("Prometheus metrics initialized")
    
    def increment_predictions(self, model_id: str, status: str = "success"):
        """Increment prediction counter."""
        if PROMETHEUS_AVAILABLE:
            self.predictions_total.labels(model_id=model_id, status=status).inc()
    
    def increment_prediction_errors(self, model_id: str, error_type: str):
        """Increment prediction error counter."""
        if PROMETHEUS_AVAILABLE:
            self.prediction_errors_total.labels(model_id=model_id, error_type=error_type).inc()
    
    def increment_rule_checks(self, status: str = "success"):
        """Increment rule check counter."""
        if PROMETHEUS_AVAILABLE:
            self.rule_checks_total.labels(status=status).inc()
    
    def increment_api_requests(self, method: str, endpoint: str, status_code: int):
        """Increment API request counter."""
        if PROMETHEUS_AVAILABLE:
            self.api_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()
    
    def observe_prediction_latency(self, model_id: str, latency_seconds: float):
        """Observe prediction latency."""
        if PROMETHEUS_AVAILABLE:
            self.prediction_latency_seconds.labels(model_id=model_id).observe(latency_seconds)
    
    def observe_model_load_time(self, model_id: str, load_time_seconds: float):
        """Observe model loading time."""
        if PROMETHEUS_AVAILABLE:
            self.model_load_time_seconds.labels(model_id=model_id).observe(load_time_seconds)
    
    def observe_api_request_duration(self, method: str, endpoint: str, duration_seconds: float):
        """Observe API request duration."""
        if PROMETHEUS_AVAILABLE:
            self.api_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration_seconds)
    
    def set_active_models(self, count: int):
        """Set number of active models."""
        if PROMETHEUS_AVAILABLE:
            self.active_models.set(count)
    
    def set_memory_usage(self, bytes_used: float):
        """Set memory usage in bytes."""
        if PROMETHEUS_AVAILABLE:
            self.memory_usage_bytes.set(bytes_used)
    
    def set_cpu_usage(self, percent: float):
        """Set CPU usage percentage."""
        if PROMETHEUS_AVAILABLE:
            self.cpu_usage_percent.set(percent)
    
    def set_disk_usage(self, bytes_used: float):
        """Set disk usage in bytes."""
        if PROMETHEUS_AVAILABLE:
            self.disk_usage_bytes.set(bytes_used)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry).decode('utf-8')
        else:
            return "# Prometheus metrics not available\n"


class StructuredLogger:
    """Structured logging for decisions and audit trails."""
    
    def __init__(self, decision_log_path: str = "artifacts/decision_logs.jsonl",
                 audit_log_path: str = "artifacts/audit_log.csv"):
        self.decision_log_path = Path(decision_log_path)
        self.audit_log_path = Path(audit_log_path)
        self._ensure_directories()
        self._init_audit_log()
        self._lock = threading.Lock()
    
    def _ensure_directories(self):
        """Ensure log directories exist."""
        self.decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _init_audit_log(self):
        """Initialize audit log with headers if it doesn't exist."""
        if not self.audit_log_path.exists():
            headers = [
                "timestamp", "event_type", "model_id", "rule_name",
                "input_hash", "output_data", "processing_time_ms",
                "user_id", "session_id"
            ]
            with open(self.audit_log_path, 'w') as f:
                f.write(','.join(headers) + '\n')
    
    def log_decision(self, entry: Dict[str, Any]) -> None:
        """Log a decision with structured format."""
        try:
            # Add timestamp if not present
            if 'timestamp' not in entry:
                entry['timestamp'] = datetime.now().isoformat()
            
            # Write to JSONL file
            with self._lock:
                with open(self.decision_log_path, 'a') as f:
                    f.write(json.dumps(entry, default=str) + '\n')
            
            logger.debug(f"Logged decision: {entry.get('event_type', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
    
    def log_audit(self, audit_data: Dict[str, Any]) -> None:
        """Log audit information to CSV."""
        try:
            # Prepare CSV row
            row_data = [
                audit_data.get('timestamp', datetime.now().isoformat()),
                audit_data.get('event_type', ''),
                audit_data.get('model_id', ''),
                audit_data.get('rule_name', ''),
                audit_data.get('input_hash', ''),
                json.dumps(audit_data.get('output_data', {}), default=str),
                str(audit_data.get('processing_time_ms', 0)),
                audit_data.get('user_id', ''),
                audit_data.get('session_id', '')
            ]
            
            # Write to CSV file
            with self._lock:
                with open(self.audit_log_path, 'a') as f:
                    f.write(','.join(f'"{item}"' for item in row_data) + '\n')
            
            logger.debug(f"Logged audit: {audit_data.get('event_type', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to log audit: {e}")
    
    def log_error(self, error_data: Dict[str, Any]) -> None:
        """Log error information."""
        error_entry = {
            'event_type': 'error',
            'timestamp': datetime.now().isoformat(),
            **error_data
        }
        self.log_decision(error_entry)


# Global instances
_metrics_collector: Optional[PrometheusMetrics] = None
_structured_logger: Optional[StructuredLogger] = None


def setup_prometheus_metrics() -> PrometheusMetrics:
    """Set up and return Prometheus metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = PrometheusMetrics()
    return _metrics_collector


def get_metrics_collector() -> PrometheusMetrics:
    """Get the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = setup_prometheus_metrics()
    return _metrics_collector


def setup_structured_logging() -> StructuredLogger:
    """Set up and return structured logger."""
    global _structured_logger
    if _structured_logger is None:
        try:
            config = get_monitoring_config()
            _structured_logger = StructuredLogger(
                decision_log_path=config.decision_log_path,
                audit_log_path=config.audit_log_path
            )
        except Exception:
            # Fallback to default paths
            _structured_logger = StructuredLogger()
    return _structured_logger


def get_structured_logger() -> StructuredLogger:
    """Get the global structured logger."""
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = setup_structured_logging()
    return _structured_logger


def structured_log_decision(entry: Dict[str, Any]) -> None:
    """Log decision with structured format."""
    logger_instance = get_structured_logger()
    logger_instance.log_decision(entry)


def track_request(endpoint: str = None, model_id: str = None):
    """Decorator to track API requests with metrics and logging."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            metrics = get_metrics_collector()
            logger_instance = get_structured_logger()
            
            # Extract request info
            request = None
            for arg in args:
                if hasattr(arg, 'method') and hasattr(arg, 'url'):
                    request = arg
                    break
            
            method = request.method if request else "UNKNOWN"
            path = endpoint or (request.url.path if request else func.__name__)
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Track success metrics
                duration = time.time() - start_time
                metrics.observe_api_request_duration(method, path, duration)
                metrics.increment_api_requests(method, path, 200)
                
                # Log decision
                log_entry = {
                    'event_type': 'api_request',
                    'method': method,
                    'endpoint': path,
                    'status': 'success',
                    'duration_ms': duration * 1000,
                    'model_id': model_id
                }
                logger_instance.log_decision(log_entry)
                
                return result
                
            except Exception as e:
                # Track error metrics
                duration = time.time() - start_time
                metrics.observe_api_request_duration(method, path, duration)
                metrics.increment_api_requests(method, path, 500)
                
                # Log error
                error_entry = {
                    'event_type': 'api_error',
                    'method': method,
                    'endpoint': path,
                    'error': str(e),
                    'duration_ms': duration * 1000,
                    'model_id': model_id
                }
                logger_instance.log_error(error_entry)
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            metrics = get_metrics_collector()
            logger_instance = get_structured_logger()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Track success metrics
                duration = time.time() - start_time
                func_name = endpoint or func.__name__
                metrics.observe_api_request_duration("SYNC", func_name, duration)
                
                # Log decision
                log_entry = {
                    'event_type': 'function_call',
                    'function': func_name,
                    'status': 'success',
                    'duration_ms': duration * 1000,
                    'model_id': model_id
                }
                logger_instance.log_decision(log_entry)
                
                return result
                
            except Exception as e:
                # Track error
                duration = time.time() - start_time
                
                # Log error
                error_entry = {
                    'event_type': 'function_error',
                    'function': func.__name__,
                    'error': str(e),
                    'duration_ms': duration * 1000,
                    'model_id': model_id
                }
                logger_instance.log_error(error_entry)
                
                raise
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and 'async' in str(func.__code__):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def update_system_metrics():
    """Update system resource metrics."""
    try:
        import psutil
        
        metrics = get_metrics_collector()
        
        # Memory usage
        memory_info = psutil.virtual_memory()
        metrics.set_memory_usage(memory_info.used)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.set_cpu_usage(cpu_percent)
        
        # Disk usage
        disk_info = psutil.disk_usage('.')
        metrics.set_disk_usage(disk_info.used)
        
    except ImportError:
        logger.warning("psutil not available for system metrics")
    except Exception as e:
        logger.error(f"Failed to update system metrics: {e}")


if __name__ == "__main__":
    # Test monitoring utilities
    try:
        print("Testing monitoring utilities...")
        
        # Test metrics collector
        metrics = setup_prometheus_metrics()
        
        # Test some metrics
        metrics.increment_predictions("test_model", "success")
        metrics.observe_prediction_latency("test_model", 0.05)
        metrics.set_active_models(2)
        
        print("✅ Metrics collector test passed")
        
        # Test structured logger
        logger_instance = setup_structured_logging()
        
        # Test decision logging
        test_decision = {
            'event_type': 'test_prediction',
            'model_id': 'test_model',
            'action': 1,
            'confidence': 0.85
        }
        logger_instance.log_decision(test_decision)
        
        # Test audit logging
        test_audit = {
            'event_type': 'test_audit',
            'model_id': 'test_model',
            'processing_time_ms': 15.5
        }
        logger_instance.log_audit(test_audit)
        
        print("✅ Structured logger test passed")
        
        # Test decorator
        @track_request(endpoint="/test", model_id="test_model")
        def test_function():
            time.sleep(0.01)  # Simulate work
            return "success"
        
        result = test_function()
        print(f"✅ Decorator test passed: {result}")
        
        # Test metrics generation
        metrics_output = metrics.get_metrics()
        print(f"✅ Metrics output generated ({len(metrics_output)} chars)")
        
        # Update system metrics
        update_system_metrics()
        print("✅ System metrics updated")
        
        print("✅ Monitoring utilities test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()