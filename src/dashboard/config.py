"""Dashboard configuration settings."""

import os
from pathlib import Path

# Dashboard settings
DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', 8501))
DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', 'localhost')
DASHBOARD_TITLE = "Market Anomaly Detection Dashboard"

# Data settings
MAX_DATA_POINTS = int(os.getenv('MAX_DATA_POINTS', 1000))
UPDATE_INTERVAL = int(os.getenv('UPDATE_INTERVAL', 5))  # seconds
CACHE_TTL = int(os.getenv('CACHE_TTL', 300))  # seconds

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
EXPLANATIONS_DIR = ARTIFACTS_DIR / "explanations"

# Visualization settings
PLOT_HEIGHT = 400
PLOT_WIDTH = 800
COLOR_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# Anomaly detection settings
ANOMALY_THRESHOLD = 0.5
RISK_LEVELS = {
    'low': (0.0, 0.3),
    'medium': (0.3, 0.7),
    'high': (0.7, 1.0)
}

# Model settings
SUPPORTED_MODELS = ['PPO', 'SAC', 'RandomForest', 'IsolationForest']
EXPLANATION_METHODS = ['shap', 'lime', 'rule']

# Real-time settings
STREAM_BUFFER_SIZE = 100
STREAM_UPDATE_FREQUENCY = 1.0  # seconds
ENABLE_REAL_TIME = True

# Dashboard layout
SIDEBAR_WIDTH = 300
MAIN_CONTENT_WIDTH = 1200

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
ENABLE_DEBUG = os.getenv('ENABLE_DEBUG', 'false').lower() == 'true'

# Performance settings
ENABLE_CACHING = True
MAX_CACHE_SIZE = 100  # MB
LAZY_LOADING = True

# Security settings (for production)
ENABLE_AUTH = os.getenv('ENABLE_AUTH', 'false').lower() == 'true'
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')

# Feature flags
ENABLE_EXPLAINABILITY = True
ENABLE_MODEL_COMPARISON = True
ENABLE_ALERTS = True
ENABLE_EXPORT = True