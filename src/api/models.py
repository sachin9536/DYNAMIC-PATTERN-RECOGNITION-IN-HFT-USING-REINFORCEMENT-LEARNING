"""Pydantic models for API request/response validation."""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
import numpy as np


# Request Models
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    sequence: List[List[float]] = Field(..., description="Input sequence data")
    model_id: str = Field(..., description="ID of the model to use for prediction")
    include_explanation: bool = Field(False, description="Whether to include explanation")
    
    @validator('sequence')
    def validate_sequence(cls, v):
        if not v:
            raise ValueError("Sequence cannot be empty")
        if not all(isinstance(row, list) for row in v):
            raise ValueError("Sequence must be a list of lists")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "sequence": [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]],
                "model_id": "ppo_v1.2.0",
                "include_explanation": True
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction endpoint."""
    sequences: List[List[List[float]]] = Field(..., description="List of input sequences")
    model_id: str = Field(..., description="ID of the model to use for prediction")
    include_explanations: bool = Field(False, description="Whether to include explanations")
    
    @validator('sequences')
    def validate_sequences(cls, v):
        if not v:
            raise ValueError("Sequences cannot be empty")
        if len(v) > 100:  # Limit batch size
            raise ValueError("Batch size cannot exceed 100")
        return v


class RuleCheckRequest(BaseModel):
    """Request model for rule checking endpoint."""
    stats: Dict[str, float] = Field(..., description="Market statistics to check")
    thresholds: Optional[Dict[str, float]] = Field(None, description="Custom thresholds")
    
    @validator('stats')
    def validate_stats(cls, v):
        if not v:
            raise ValueError("Stats cannot be empty")
        # Check for required fields
        required_fields = ['price', 'volume', 'volatility']
        missing_fields = [field for field in required_fields if field not in v]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "stats": {
                    "price": 100.5,
                    "volume": 1500,
                    "volatility": 0.02,
                    "returns": 0.001
                },
                "thresholds": {
                    "volatility_threshold": 0.05,
                    "volume_threshold": 2000
                }
            }
        }


class ModelLoadRequest(BaseModel):
    """Request model for loading a model."""
    model_id: str = Field(..., description="ID of the model to load")
    force_reload: bool = Field(False, description="Force reload if already loaded")


# Response Models
class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    action: int = Field(..., description="Predicted action")
    scores: List[float] = Field(..., description="Action probability scores")
    confidence: float = Field(..., description="Prediction confidence")
    explanation_hash: Optional[str] = Field(None, description="Hash of explanation data")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_id: str = Field(..., description="ID of the model used")
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "action": 1,
                "scores": [0.1, 0.8, 0.1],
                "confidence": 0.85,
                "explanation_hash": "abc123def",
                "processing_time_ms": 15.5,
                "model_id": "ppo_v1.2.0",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction endpoint."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_processing_time_ms: float = Field(..., description="Total processing time")
    batch_size: int = Field(..., description="Number of predictions in batch")


class RuleCheckResponse(BaseModel):
    """Response model for rule checking endpoint."""
    rule_flags: Dict[str, bool] = Field(..., description="Rule trigger flags")
    triggered_rules: List[str] = Field(..., description="List of triggered rule names")
    anomaly_score: float = Field(..., description="Overall anomaly score")
    explanation_text: str = Field(..., description="Human-readable explanation")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: str = Field(..., description="Check timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "rule_flags": {
                    "high_volatility": True,
                    "volume_spike": False,
                    "price_anomaly": True
                },
                "triggered_rules": ["high_volatility", "price_anomaly"],
                "anomaly_score": 0.75,
                "explanation_text": "High volatility and price anomaly detected",
                "processing_time_ms": 5.2,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str = Field(..., description="System health status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="System version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0",
                "uptime_seconds": 3600.0
            }
        }


class SystemStatusResponse(BaseModel):
    """Response model for detailed system status."""
    status: str = Field(..., description="Overall system status")
    timestamp: str = Field(..., description="Status check timestamp")
    version: str = Field(..., description="System version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    loaded_models: List[str] = Field(..., description="List of loaded model IDs")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    disk_usage_mb: float = Field(..., description="Disk usage in MB")
    components: Dict[str, str] = Field(..., description="Component status")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0",
                "uptime_seconds": 3600.0,
                "loaded_models": ["ppo_v1.2.0", "sac_v1.1.0"],
                "memory_usage_mb": 512.3,
                "cpu_usage_percent": 15.2,
                "disk_usage_mb": 1024.5,
                "components": {
                    "model_manager": "running",
                    "rule_engine": "running",
                    "metrics_collector": "running"
                }
            }
        }


class ModelInfo(BaseModel):
    """Model information structure."""
    model_id: str = Field(..., description="Model identifier")
    algorithm: str = Field(..., description="Model algorithm")
    file_path: str = Field(..., description="Path to model file")
    file_size_mb: float = Field(..., description="File size in MB")
    status: str = Field(..., description="Model status (available/loaded/error)")
    created_at: Optional[str] = Field(None, description="Model creation timestamp")
    modified_at: str = Field(..., description="Last modification timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ModelsListResponse(BaseModel):
    """Response model for models list endpoint."""
    models: List[ModelInfo] = Field(..., description="List of available models")
    total_count: int = Field(..., description="Total number of models")
    loaded_count: int = Field(..., description="Number of loaded models")
    
    class Config:
        schema_extra = {
            "example": {
                "models": [
                    {
                        "model_id": "ppo_v1.2.0",
                        "algorithm": "PPO",
                        "file_path": "artifacts/models/ppo_v1.2.0.zip",
                        "file_size_mb": 45.2,
                        "status": "loaded",
                        "created_at": "2024-01-10T08:00:00Z",
                        "modified_at": "2024-01-10T08:00:00Z",
                        "metadata": {"accuracy": 0.87}
                    }
                ],
                "total_count": 3,
                "loaded_count": 1
            }
        }


class ModelLoadResponse(BaseModel):
    """Response model for model loading."""
    model_id: str = Field(..., description="Loaded model ID")
    status: str = Field(..., description="Load operation status")
    message: str = Field(..., description="Status message")
    loading_time_ms: float = Field(..., description="Loading time in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": "ppo_v1.2.0",
                "status": "success",
                "message": "Model loaded successfully",
                "loading_time_ms": 1250.5
            }
        }


class ModelUnloadResponse(BaseModel):
    """Response model for model unloading."""
    model_id: str = Field(..., description="Unloaded model ID")
    status: str = Field(..., description="Unload operation status")
    message: str = Field(..., description="Status message")


class RulesSummaryResponse(BaseModel):
    """Response model for rules summary."""
    total_rules: int = Field(..., description="Total number of rules")
    rule_types: List[str] = Field(..., description="Types of rules available")
    rule_names: List[str] = Field(..., description="Names of all rules")
    thresholds: Dict[str, float] = Field(..., description="Current rule thresholds")
    
    class Config:
        schema_extra = {
            "example": {
                "total_rules": 5,
                "rule_types": ["volatility", "volume", "price"],
                "rule_names": ["high_volatility", "volume_spike", "price_anomaly"],
                "thresholds": {
                    "volatility_threshold": 0.05,
                    "volume_threshold": 2000,
                    "price_threshold": 0.02
                }
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: Dict[str, Any] = Field(..., description="Error information")
    
    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "code": "MODEL_NOT_FOUND",
                    "message": "Model 'invalid_model' not found",
                    "details": {
                        "available_models": ["ppo_v1.2.0", "sac_v1.1.0"],
                        "request_id": "req_123456789"
                    },
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            }
        }


# Utility functions for model conversion
def numpy_to_list(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    return obj


def create_error_response(code: str, message: str, details: Optional[Dict[str, Any]] = None) -> ErrorResponse:
    """Create a standardized error response."""
    error_data = {
        "code": code,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    
    if details:
        error_data["details"] = details
    
    return ErrorResponse(error=error_data)


if __name__ == "__main__":
    # Test Pydantic models
    try:
        print("Testing Pydantic models...")
        
        # Test PredictionRequest
        pred_req = PredictionRequest(
            sequence=[[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]],
            model_id="test_model",
            include_explanation=True
        )
        print(f"Prediction request: {pred_req.dict()}")
        
        # Test PredictionResponse
        pred_resp = PredictionResponse(
            action=1,
            scores=[0.1, 0.8, 0.1],
            confidence=0.85,
            explanation_hash="abc123",
            processing_time_ms=15.5,
            model_id="test_model",
            timestamp=datetime.now().isoformat()
        )
        print(f"Prediction response: {pred_resp.dict()}")
        
        # Test RuleCheckRequest
        rule_req = RuleCheckRequest(
            stats={"price": 100.5, "volume": 1500, "volatility": 0.02},
            thresholds={"volatility_threshold": 0.05}
        )
        print(f"Rule check request: {rule_req.dict()}")
        
        # Test error response creation
        error_resp = create_error_response(
            code="TEST_ERROR",
            message="This is a test error",
            details={"test_field": "test_value"}
        )
        print(f"Error response: {error_resp.dict()}")
        
        print("✅ Pydantic models test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()