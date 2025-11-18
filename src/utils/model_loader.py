"""Model loading and management utilities."""

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import numpy as np

try:
    from src.utils.logger import get_logger
    from src.utils.interfaces import ModelInterface, PredictionResult
    from src.utils.config_manager import get_config_manager
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Fallback imports
    class ModelInterface:
        def predict(self, observation): pass
        def load(self, model_path): pass
        def get_info(self): pass
    
    class PredictionResult:
        def __init__(self, action, scores, confidence, explanation_hash=None, processing_time_ms=0.0):
            self.action = action
            self.scores = scores
            self.confidence = confidence
            self.explanation_hash = explanation_hash
            self.processing_time_ms = processing_time_ms


class ModelWrapper(ModelInterface):
    """Wrapper for different model types."""
    
    def __init__(self, model_path: str, model_type: str = "auto"):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.metadata = {}
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the model from file."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Determine model type if auto
            if self.model_type == "auto":
                self.model_type = self._detect_model_type()
            
            # Load based on type
            if self.model_type == "stable_baselines3":
                self._load_sb3_model()
            elif self.model_type == "sklearn":
                self._load_sklearn_model()
            elif self.model_type == "pytorch":
                self._load_pytorch_model()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Load metadata
            self._load_metadata()
            
            logger.info(f"Successfully loaded {self.model_type} model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise
    
    def _detect_model_type(self) -> str:
        """Detect model type from file extension and content."""
        file_ext = Path(self.model_path).suffix.lower()
        
        if file_ext == '.zip':
            # Likely Stable-Baselines3 model
            return "stable_baselines3"
        elif file_ext in ['.pkl', '.pickle']:
            # Likely scikit-learn model
            return "sklearn"
        elif file_ext in ['.pt', '.pth']:
            # Likely PyTorch model
            return "pytorch"
        else:
            # Default to stable_baselines3 for unknown extensions
            return "stable_baselines3"
    
    def _load_sb3_model(self) -> None:
        """Load Stable-Baselines3 model."""
        try:
            from stable_baselines3 import PPO, SAC, A2C, DQN
            
            # Try different algorithms
            algorithms = [PPO, SAC, A2C, DQN]
            
            for algo in algorithms:
                try:
                    self.model = algo.load(self.model_path)
                    self.metadata['algorithm'] = algo.__name__
                    break
                except Exception:
                    continue
            
            if self.model is None:
                raise ValueError("Could not load model with any supported algorithm")
                
        except ImportError:
            logger.warning("Stable-Baselines3 not available, using fallback")
            self._load_fallback_model()
    
    def _load_sklearn_model(self) -> None:
        """Load scikit-learn model."""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.metadata['algorithm'] = type(self.model).__name__
        except Exception as e:
            logger.error(f"Failed to load sklearn model: {e}")
            self._load_fallback_model()
    
    def _load_pytorch_model(self) -> None:
        """Load PyTorch model."""
        try:
            import torch
            self.model = torch.load(self.model_path, map_location='cpu')
            self.metadata['algorithm'] = 'PyTorch'
        except ImportError:
            logger.warning("PyTorch not available, using fallback")
            self._load_fallback_model()
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            self._load_fallback_model()
    
    def _load_fallback_model(self) -> None:
        """Load fallback dummy model."""
        logger.warning("Using fallback dummy model")
        self.model = DummyModel()
        self.metadata['algorithm'] = 'Dummy'
    
    def _load_metadata(self) -> None:
        """Load model metadata."""
        metadata_path = Path(self.model_path).with_suffix('.json')
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    file_metadata = json.load(f)
                self.metadata.update(file_metadata)
            except Exception as e:
                logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        
        # Add basic metadata
        self.metadata.update({
            'model_path': str(self.model_path),
            'model_type': self.model_type,
            'file_size_mb': os.path.getsize(self.model_path) / (1024 * 1024),
            'loaded_at': datetime.now().isoformat()
        })
    
    def predict(self, observation: Union[np.ndarray, List[float]]) -> PredictionResult:
        """Make prediction using the loaded model."""
        try:
            start_time = datetime.now()
            
            # Convert observation to appropriate format
            if isinstance(observation, list):
                observation = np.array(observation)
            
            # Make prediction based on model type
            if self.model_type == "stable_baselines3":
                action, _states = self.model.predict(observation, deterministic=True)
                if isinstance(action, np.ndarray):
                    action = int(action[0]) if action.size == 1 else int(action)
                scores = [0.0, 0.0, 1.0]  # Placeholder scores
                confidence = 0.85  # Placeholder confidence
                
            elif self.model_type == "sklearn":
                if hasattr(self.model, 'predict_proba'):
                    scores = self.model.predict_proba(observation.reshape(1, -1))[0].tolist()
                    action = int(np.argmax(scores))
                    confidence = max(scores)
                else:
                    action = int(self.model.predict(observation.reshape(1, -1))[0])
                    scores = [0.0, 0.0, 1.0]
                    confidence = 0.8
                    
            else:  # Fallback or other types
                action = np.random.randint(0, 3)
                scores = np.random.dirichlet([1, 1, 1]).tolist()
                confidence = max(scores)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Generate explanation hash
            explanation_hash = hashlib.md5(str(observation).encode()).hexdigest()[:8]
            
            return PredictionResult(
                action=action,
                scores=scores,
                confidence=confidence,
                explanation_hash=explanation_hash,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return fallback prediction
            return PredictionResult(
                action=0,
                scores=[1.0, 0.0, 0.0],
                confidence=0.5,
                explanation_hash="error",
                processing_time_ms=0.0
            )
    
    def load(self, model_path: str) -> None:
        """Load model from new path."""
        self.model_path = model_path
        self._load_model()
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.metadata.copy()


class DummyModel:
    """Dummy model for fallback purposes."""
    
    def predict(self, observation, deterministic=True):
        """Make random prediction."""
        action = np.random.randint(0, 3)
        return action, None


class ModelManager:
    """Manages multiple models with caching."""
    
    def __init__(self, models_dir: str = "artifacts/models"):
        self.models_dir = Path(models_dir)
        self.models_cache: Dict[str, ModelWrapper] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self._scan_models()
    
    def _scan_models(self) -> None:
        """Scan models directory for available models."""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            return
        
        # Look for model files
        model_extensions = ['.zip', '.pkl', '.pickle', '.pt', '.pth']
        
        for model_file in self.models_dir.iterdir():
            if model_file.suffix.lower() in model_extensions:
                model_id = model_file.stem
                
                # Load metadata if available
                metadata_file = model_file.with_suffix('.json')
                metadata = {}
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to load metadata for {model_id}: {e}")
                
                # Add basic info
                metadata.update({
                    'model_id': model_id,
                    'file_path': str(model_file),
                    'file_size_mb': model_file.stat().st_size / (1024 * 1024),
                    'modified_at': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                    'status': 'available'
                })
                
                self.model_metadata[model_id] = metadata
        
        logger.info(f"Found {len(self.model_metadata)} models in {self.models_dir}")
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        return list(self.model_metadata.values())
    
    def load_model(self, model_id: str) -> ModelWrapper:
        """Load a model by ID."""
        if model_id in self.models_cache:
            logger.info(f"Model {model_id} already loaded from cache")
            return self.models_cache[model_id]
        
        if model_id not in self.model_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        model_path = self.model_metadata[model_id]['file_path']
        
        try:
            model_wrapper = ModelWrapper(model_path)
            self.models_cache[model_id] = model_wrapper
            self.model_metadata[model_id]['status'] = 'loaded'
            
            logger.info(f"Successfully loaded model {model_id}")
            return model_wrapper
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            self.model_metadata[model_id]['status'] = 'error'
            raise
    
    def unload_model(self, model_id: str) -> None:
        """Unload a model from cache."""
        if model_id in self.models_cache:
            del self.models_cache[model_id]
            if model_id in self.model_metadata:
                self.model_metadata[model_id]['status'] = 'available'
            logger.info(f"Unloaded model {model_id}")
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get model information."""
        if model_id not in self.model_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        info = self.model_metadata[model_id].copy()
        
        # Add runtime info if loaded
        if model_id in self.models_cache:
            model_info = self.models_cache[model_id].get_info()
            info.update(model_info)
        
        return info
    
    def validate_model(self, model_path: str) -> bool:
        """Validate if a model file is loadable."""
        try:
            # Try to create a wrapper (but don't cache it)
            wrapper = ModelWrapper(model_path)
            return True
        except Exception as e:
            logger.error(f"Model validation failed for {model_path}: {e}")
            return False
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded model IDs."""
        return list(self.models_cache.keys())
    
    def clear_cache(self) -> None:
        """Clear all loaded models from cache."""
        self.models_cache.clear()
        for model_id in self.model_metadata:
            self.model_metadata[model_id]['status'] = 'available'
        logger.info("Cleared model cache")


if __name__ == "__main__":
    # Test model manager
    try:
        print("Testing Model Manager...")
        
        # Create model manager
        manager = ModelManager()
        
        # List available models
        models = manager.list_available_models()
        print(f"Available models: {len(models)}")
        
        for model in models:
            print(f"  - {model['model_id']}: {model.get('algorithm', 'Unknown')} "
                  f"({model['file_size_mb']:.1f} MB)")
        
        # Test with dummy data if models available
        if models:
            model_id = models[0]['model_id']
            print(f"\nTesting model: {model_id}")
            
            try:
                # Load model
                model_wrapper = manager.load_model(model_id)
                
                # Test prediction
                dummy_observation = np.random.randn(10)
                result = model_wrapper.predict(dummy_observation)
                print(f"Prediction result: action={result.action}, confidence={result.confidence:.3f}")
                
                # Get model info
                info = manager.get_model_info(model_id)
                print(f"Model info keys: {list(info.keys())}")
                
            except Exception as e:
                print(f"Model test failed: {e}")
        
        print("✅ Model manager test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()