"""Main FastAPI application for the market anomaly detection system."""

import os
import sys
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from contextlib import asynccontextmanager

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.utils.logger import get_logger
    from src.utils.config_manager import get_api_config, get_monitoring_config
    from src.utils.model_loader import ModelManager
    from src.api.models import *
    logger = get_logger(__name__)
except ImportError as e:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print(f"Import warning: {e}")
    
    # Fallback imports
    class ModelManager:
        def __init__(self): pass
        def list_available_models(self): return []
        def load_model(self, model_id): pass
        def get_model_info(self, model_id): return {}


# Global state
app_start_time = datetime.now()
model_manager: Optional[ModelManager] = None
rule_engine = None
metrics_collector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting FastAPI application...")
    
    global model_manager, rule_engine, metrics_collector
    
    try:
        # Initialize model manager
        model_manager = ModelManager()
        logger.info("Model manager initialized")
        
        # Initialize rule engine (if available)
        try:
            from src.explainability.rule_based import MarketAnomalyRules
            rule_engine = MarketAnomalyRules()
            logger.info("Rule engine initialized")
        except ImportError:
            logger.warning("Rule engine not available")
        
        # Initialize metrics collector
        try:
            from src.utils.monitoring import setup_prometheus_metrics
            metrics_collector = setup_prometheus_metrics()
            logger.info("Metrics collector initialized")
        except ImportError:
            logger.warning("Metrics collector not available")
        
        logger.info("FastAPI application startup complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    
    # Cleanup resources
    if model_manager:
        model_manager.clear_cache()
    
    logger.info("FastAPI application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Market Anomaly Detection API",
    description="REST API for market anomaly detection system with ML models and rule-based analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
try:
    api_config = get_api_config()
    
    # CORS middleware
    if api_config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=api_config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Trusted host middleware (for production)
    if not api_config.reload:  # Only in production
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", api_config.host]
        )
        
except Exception as e:
    logger.warning(f"Failed to load API config, using defaults: {e}")


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Log request
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            code=f"HTTP_{exc.status_code}",
            message=exc.detail,
            details={"path": str(request.url.path)}
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=create_error_response(
            code="INTERNAL_SERVER_ERROR",
            message="An internal server error occurred",
            details={"path": str(request.url.path)}
        ).dict()
    )


# Dependency functions
def get_model_manager() -> ModelManager:
    """Get the model manager instance."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    return model_manager


def get_rule_engine():
    """Get the rule engine instance."""
    if rule_engine is None:
        raise HTTPException(status_code=503, detail="Rule engine not available")
    return rule_engine


# Health and status endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime_seconds=uptime
    )


@app.get("/status", response_model=SystemStatusResponse)
async def system_status(mgr: ModelManager = Depends(get_model_manager)):
    """Detailed system status endpoint."""
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    # Get system metrics
    memory_info = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    disk_info = psutil.disk_usage('.')
    
    # Get loaded models
    loaded_models = mgr.get_loaded_models()
    
    # Component status
    components = {
        "model_manager": "running",
        "rule_engine": "running" if rule_engine else "unavailable",
        "metrics_collector": "running" if metrics_collector else "unavailable"
    }
    
    return SystemStatusResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        uptime_seconds=uptime,
        loaded_models=loaded_models,
        memory_usage_mb=memory_info.used / (1024 * 1024),
        cpu_usage_percent=cpu_percent,
        disk_usage_mb=disk_info.used / (1024 * 1024),
        components=components
    )


# Model management endpoints
@app.get("/models", response_model=ModelsListResponse)
async def list_models(mgr: ModelManager = Depends(get_model_manager)):
    """List all available models."""
    try:
        models = mgr.list_available_models()
        loaded_models = mgr.get_loaded_models()
        
        # Convert to ModelInfo objects
        model_infos = []
        for model_data in models:
            model_info = ModelInfo(
                model_id=model_data['model_id'],
                algorithm=model_data.get('algorithm', 'Unknown'),
                file_path=model_data['file_path'],
                file_size_mb=model_data['file_size_mb'],
                status=model_data['status'],
                created_at=model_data.get('created_at'),
                modified_at=model_data['modified_at'],
                metadata=model_data.get('metadata', {})
            )
            model_infos.append(model_info)
        
        return ModelsListResponse(
            models=model_infos,
            total_count=len(models),
            loaded_count=len(loaded_models)
        )
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@app.post("/models/{model_id}/load", response_model=ModelLoadResponse)
async def load_model(model_id: str, request: ModelLoadRequest, mgr: ModelManager = Depends(get_model_manager)):
    """Load a specific model."""
    try:
        start_time = time.time()
        
        # Check if already loaded
        if model_id in mgr.get_loaded_models() and not request.force_reload:
            return ModelLoadResponse(
                model_id=model_id,
                status="already_loaded",
                message="Model is already loaded",
                loading_time_ms=0.0
            )
        
        # Load the model
        mgr.load_model(model_id)
        loading_time = (time.time() - start_time) * 1000
        
        return ModelLoadResponse(
            model_id=model_id,
            status="success",
            message="Model loaded successfully",
            loading_time_ms=loading_time
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.delete("/models/{model_id}/unload", response_model=ModelUnloadResponse)
async def unload_model(model_id: str, mgr: ModelManager = Depends(get_model_manager)):
    """Unload a specific model."""
    try:
        mgr.unload_model(model_id)
        
        return ModelUnloadResponse(
            model_id=model_id,
            status="success",
            message="Model unloaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to unload model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")


# Prediction endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, mgr: ModelManager = Depends(get_model_manager)):
    """Make a prediction using the specified model."""
    try:
        start_time = time.time()
        
        # Load model if not already loaded
        if request.model_id not in mgr.get_loaded_models():
            mgr.load_model(request.model_id)
        
        # Get model
        model_wrapper = mgr.models_cache[request.model_id]
        
        # Convert sequence to numpy array
        import numpy as np
        sequence_array = np.array(request.sequence)
        
        # Make prediction
        result = model_wrapper.predict(sequence_array)
        
        # Calculate total processing time
        total_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            action=result.action,
            scores=result.scores,
            confidence=result.confidence,
            explanation_hash=result.explanation_hash,
            processing_time_ms=total_time,
            model_id=request.model_id,
            timestamp=datetime.now().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest, mgr: ModelManager = Depends(get_model_manager)):
    """Make batch predictions using the specified model."""
    try:
        start_time = time.time()
        
        # Load model if not already loaded
        if request.model_id not in mgr.get_loaded_models():
            mgr.load_model(request.model_id)
        
        # Get model
        model_wrapper = mgr.models_cache[request.model_id]
        
        # Make predictions for each sequence
        predictions = []
        for sequence in request.sequences:
            import numpy as np
            sequence_array = np.array(sequence)
            result = model_wrapper.predict(sequence_array)
            
            pred_response = PredictionResponse(
                action=result.action,
                scores=result.scores,
                confidence=result.confidence,
                explanation_hash=result.explanation_hash,
                processing_time_ms=result.processing_time_ms,
                model_id=request.model_id,
                timestamp=datetime.now().isoformat()
            )
            predictions.append(pred_response)
        
        # Calculate total processing time
        total_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time_ms=total_time,
            batch_size=len(predictions)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# Rules endpoints
@app.post("/rules/check", response_model=RuleCheckResponse)
async def check_rules(request: RuleCheckRequest, rules: Any = Depends(get_rule_engine)):
    """Check rules against market statistics."""
    try:
        start_time = time.time()
        
        # Convert stats to the format expected by rule engine
        feature_names = list(request.stats.keys())
        observation = list(request.stats.values())
        
        # Check rules
        result = rules.explain_observation(observation, feature_names)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return RuleCheckResponse(
            rule_flags={rule: True for rule in result.get('triggered_rules', [])},
            triggered_rules=result.get('triggered_rules', []),
            anomaly_score=result.get('anomaly_score', 0.0),
            explanation_text=result.get('explanation_text', ''),
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Rule check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rule check failed: {str(e)}")


@app.get("/rules/summary", response_model=RulesSummaryResponse)
async def rules_summary(rules: Any = Depends(get_rule_engine)):
    """Get summary of available rules."""
    try:
        summary = rules.get_rule_summary()
        
        return RulesSummaryResponse(
            total_rules=summary.get('total_rules', 0),
            rule_types=summary.get('rule_types', []),
            rule_names=summary.get('rule_names', []),
            thresholds=summary.get('thresholds', {})
        )
        
    except Exception as e:
        logger.error(f"Failed to get rules summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get rules summary")


# Metrics endpoint
@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint."""
    try:
        if metrics_collector:
            from prometheus_client import generate_latest
            return generate_latest()
        else:
            # Return basic metrics if prometheus not available
            uptime = (datetime.now() - app_start_time).total_seconds()
            return f"# HELP uptime_seconds System uptime in seconds\n# TYPE uptime_seconds gauge\nuptime_seconds {uptime}\n"
            
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return "# Metrics unavailable\n"


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Market Anomaly Detection API",
        "version": "1.0.0",
        "description": "REST API for market anomaly detection system",
        "docs_url": "/docs",
        "health_url": "/health",
        "status_url": "/status",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    try:
        api_config = get_api_config()
        
        uvicorn.run(
            "src.api.app:app",
            host=api_config.host,
            port=api_config.port,
            reload=api_config.reload,
            workers=api_config.workers if not api_config.reload else 1,
            log_level="info"
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        # Fallback configuration
        uvicorn.run(
            "src.api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )