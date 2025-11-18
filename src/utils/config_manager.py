"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    title: str = "Market Anomaly Detection Dashboard"
    port: int = 8501
    host: str = "0.0.0.0"
    auto_refresh: bool = True
    refresh_interval: int = 5
    theme: str = "light"
    max_data_points: int = 1000
    enable_real_time: bool = True


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    enable_cors: bool = True
    cors_origins: list = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:8501", "http://127.0.0.1:8501"]


@dataclass
class SimulationConfig:
    """Simulation configuration."""
    default_duration: int = 3600
    update_frequency: float = 1.0
    max_concurrent: int = 5
    buffer_size: int = 100


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    enable_prometheus: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    metrics_port: int = 9090
    enable_audit_log: bool = True
    audit_log_path: str = "artifacts/audit_log.csv"
    decision_log_path: str = "artifacts/decision_logs.jsonl"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: str = "development"
    debug: bool = True
    enable_auth: bool = False
    secret_key: str = "dev-key-change-in-production"


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    enable_caching: bool = True
    cache_ttl: int = 300
    max_cache_size_mb: int = 100
    lazy_loading: bool = True


class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self._config_data = None
        self._load_config()
    
    def _find_config_file(self) -> str:
        """Find the configuration file."""
        possible_paths = [
            "configs/config.yaml",
            "../configs/config.yaml",
            "../../configs/config.yaml",
            os.path.join(os.path.dirname(__file__), "../../configs/config.yaml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # If no config found, create a default one
        logger.warning("No config file found, using defaults")
        return None
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self._config_data = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load config from {self.config_path}: {e}")
                self._config_data = {}
        else:
            self._config_data = {}
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a configuration section."""
        return self._config_data.get(section, {})
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation."""
        keys = key.split('.')
        value = self._config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_dashboard_config(self) -> DashboardConfig:
        """Get dashboard configuration."""
        dashboard_data = self.get_section('dashboard')
        return DashboardConfig(**dashboard_data)
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration."""
        api_data = self.get_section('api')
        return APIConfig(**api_data)
    
    def get_simulation_config(self) -> SimulationConfig:
        """Get simulation configuration."""
        sim_data = self.get_section('simulation')
        return SimulationConfig(**sim_data)
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        mon_data = self.get_section('monitoring')
        return MonitoringConfig(**mon_data)
    
    def get_deployment_config(self) -> DeploymentConfig:
        """Get deployment configuration."""
        deploy_data = self.get_section('deployment')
        return DeploymentConfig(**deploy_data)
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration."""
        perf_data = self.get_section('performance')
        return PerformanceConfig(**perf_data)
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> None:
        """Update configuration section."""
        if section not in self._config_data:
            self._config_data[section] = {}
        
        self._config_data[section].update(updates)
        logger.info(f"Updated configuration section: {section}")
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save configuration to file."""
        save_path = path or self.config_path
        if save_path:
            try:
                with open(save_path, 'w') as f:
                    yaml.dump(self._config_data, f, default_flow_style=False)
                logger.info(f"Saved configuration to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save config to {save_path}: {e}")


# Global configuration manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_dashboard_config() -> DashboardConfig:
    """Get dashboard configuration."""
    return get_config_manager().get_dashboard_config()


def get_api_config() -> APIConfig:
    """Get API configuration."""
    return get_config_manager().get_api_config()


def get_simulation_config() -> SimulationConfig:
    """Get simulation configuration."""
    return get_config_manager().get_simulation_config()


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration."""
    return get_config_manager().get_monitoring_config()


def get_deployment_config() -> DeploymentConfig:
    """Get deployment configuration."""
    return get_config_manager().get_deployment_config()


def get_performance_config() -> PerformanceConfig:
    """Get performance configuration."""
    return get_config_manager().get_performance_config()


if __name__ == "__main__":
    # Test configuration manager
    try:
        print("Testing Configuration Manager...")
        
        config_mgr = ConfigManager()
        
        # Test getting different configurations
        dashboard_cfg = config_mgr.get_dashboard_config()
        print(f"Dashboard config: {dashboard_cfg}")
        
        api_cfg = config_mgr.get_api_config()
        print(f"API config: {api_cfg}")
        
        monitoring_cfg = config_mgr.get_monitoring_config()
        print(f"Monitoring config: {monitoring_cfg}")
        
        # Test getting specific values
        log_level = config_mgr.get_value('monitoring.log_level', 'INFO')
        print(f"Log level: {log_level}")
        
        # Test global functions
        dash_cfg = get_dashboard_config()
        print(f"Global dashboard config: {dash_cfg}")
        
        print("✅ Configuration manager test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()