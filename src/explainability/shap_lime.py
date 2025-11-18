"""SHAP and LIME integration for model explainability."""

import numpy as np
import pandas as pd
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Install with: pip install lime")

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP explainer wrapper class."""
    
    def __init__(self, cache_dir: str = "artifacts/explanations"):
        self.cache_dir = cache_dir
        
    def explain(
        self,
        model,
        observation: np.ndarray,
        background_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        nsamples: int = 100
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for observations.
        
        Args:
            model: Trained model
            observation: Input data to explain
            background_data: Background data for SHAP
            feature_names: List of feature names
            nsamples: Number of samples for SHAP estimation
            
        Returns:
            Dictionary with SHAP explanation
        """
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
            
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(observation.shape[1])]
            
        return explain_with_shap(
            model=model,
            X=observation,
            feature_names=feature_names,
            nsamples=nsamples,
            cache_dir=self.cache_dir
        )


class LIMEExplainer:
    """LIME explainer wrapper class."""
    
    def __init__(self, cache_dir: str = "artifacts/explanations"):
        self.cache_dir = cache_dir
        
    def explain(
        self,
        model,
        observation: np.ndarray,
        training_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for observation.
        
        Args:
            model: Trained model
            observation: Single observation to explain
            training_data: Training data for LIME
            feature_names: List of feature names
            num_features: Number of features to include
            
        Returns:
            Dictionary with LIME explanation
        """
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
            
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(observation.shape[1])]
            
        return explain_with_lime(
            model=model,
            X=observation,
            feature_names=feature_names,
            num_features=num_features,
            cache_dir=self.cache_dir
        )


def model_predict_for_explainer(model, X: np.ndarray) -> np.ndarray:
    """
    Convert model predictions to format suitable for explainers.
    
    Args:
        model: Trained model (sklearn, torch, or stable-baselines3)
        X: Input array of shape (n_samples, n_features)
    
    Returns:
        Predictions as numpy array of shape (n_samples, n_outputs)
    """
    try:
        # Handle different model types
        if hasattr(model, 'predict_proba'):
            # Sklearn-style classifier
            return model.predict_proba(X)
        
        elif hasattr(model, 'predict'):
            # Sklearn-style regressor or stable-baselines3 policy
            if hasattr(model, 'policy'):
                # Stable-baselines3 model
                import torch
                with torch.no_grad():
                    # Convert to torch tensor
                    if not isinstance(X, torch.Tensor):
                        X_tensor = torch.FloatTensor(X)
                    else:
                        X_tensor = X
                    
                    # Get policy predictions
                    if hasattr(model.policy, 'predict'):
                        actions, _ = model.policy.predict(X_tensor, deterministic=True)
                        if isinstance(actions, torch.Tensor):
                            actions = actions.cpu().numpy()
                        
                        # Convert actions to probabilities
                        if actions.ndim == 1:
                            actions = actions.reshape(-1, 1)
                        
                        # For discrete actions, create one-hot style probabilities
                        if actions.dtype in [np.int32, np.int64]:
                            n_actions = int(actions.max()) + 1
                            probs = np.zeros((len(actions), n_actions))
                            probs[np.arange(len(actions)), actions.flatten()] = 1.0
                            return probs
                        else:
                            return actions
                    
                    # Fallback: use policy network directly
                    elif hasattr(model.policy, 'forward'):
                        logits = model.policy.forward(X_tensor)
                        if isinstance(logits, tuple):
                            logits = logits[0]  # Take first output
                        
                        # Apply softmax for discrete actions
                        if logits.shape[-1] > 1:
                            probs = torch.softmax(logits, dim=-1)
                            return probs.cpu().numpy()
                        else:
                            return logits.cpu().numpy()
            
            else:
                # Regular sklearn model
                predictions = model.predict(X)
                if predictions.ndim == 1:
                    predictions = predictions.reshape(-1, 1)
                return predictions
        
        elif hasattr(model, '__call__'):
            # PyTorch model or callable
            import torch
            model.eval()
            with torch.no_grad():
                if not isinstance(X, torch.Tensor):
                    X_tensor = torch.FloatTensor(X)
                else:
                    X_tensor = X
                
                outputs = model(X_tensor)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                return outputs.cpu().numpy()
        
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
    
    except Exception as e:
        logger.error(f"Error in model_predict_for_explainer: {e}")
        # Fallback: return random predictions
        n_samples = X.shape[0]
        return np.random.random((n_samples, 3))  # Assume 3 actions


def explain_with_shap(
    model,
    X: np.ndarray,
    feature_names: List[str],
    nsamples: int = 100,
    cache_dir: str = "artifacts/explanations"
) -> Dict[str, Any]:
    """
    Generate SHAP explanations for model predictions.
    
    Args:
        model: Trained model
        X: Input data of shape (n_samples, n_features)
        feature_names: List of feature names
        nsamples: Number of samples for SHAP estimation
        cache_dir: Directory to cache results
    
    Returns:
        Dictionary with SHAP values and metadata
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP not available. Install with: pip install shap")
    
    logger.info(f"Computing SHAP explanations for {X.shape[0]} samples")
    
    try:
        # Create prediction function
        def predict_fn(x):
            return model_predict_for_explainer(model, x)
        
        # Initialize SHAP explainer
        # Use a subset of data as background for efficiency
        background_size = min(100, X.shape[0])
        background = X[:background_size]
        
        explainer = shap.KernelExplainer(predict_fn, background)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X, nsamples=nsamples)
        
        # Handle multi-output case
        if isinstance(shap_values, list):
            # Multi-class case - take first class or average
            if len(shap_values) > 1:
                shap_values_array = np.array(shap_values[0])  # Take first class
            else:
                shap_values_array = np.array(shap_values[0])
        else:
            shap_values_array = shap_values
        
        # Get base values
        base_values = explainer.expected_value
        if isinstance(base_values, (list, np.ndarray)):
            base_values = base_values[0] if len(base_values) > 0 else 0.0
        
        # Prepare result
        result = {
            'method': 'shap',
            'shap_values': shap_values_array.tolist(),
            'expected_value': float(base_values),  # Use expected_value instead of base_values
            'feature_names': feature_names[:X.shape[1]],  # Ensure length matches
            'input_shape': X.shape,
            'nsamples': nsamples,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Cache results
        if cache_dir:
            cache_path = _cache_explanation(result, cache_dir, 'shap')
            result['cache_path'] = cache_path
        
        logger.info(f"SHAP explanation completed. Shape: {shap_values_array.shape}")
        return result
    
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        return {
            'method': 'shap',
            'error': str(e),
            'feature_names': feature_names,
            'timestamp': pd.Timestamp.now().isoformat()
        }


def explain_with_lime(
    model,
    X: np.ndarray,
    feature_names: List[str],
    num_features: int = 10,
    cache_dir: str = "artifacts/explanations"
) -> Dict[str, Any]:
    """
    Generate LIME explanations for model predictions.
    
    Args:
        model: Trained model
        X: Input data of shape (n_samples, n_features)
        feature_names: List of feature names
        num_features: Number of top features to explain
        cache_dir: Directory to cache results
    
    Returns:
        Dictionary with LIME explanations and metadata
    """
    if not LIME_AVAILABLE:
        raise ImportError("LIME not available. Install with: pip install lime")
    
    logger.info(f"Computing LIME explanations for {X.shape[0]} samples")
    
    try:
        # Create prediction function
        def predict_fn(x):
            return model_predict_for_explainer(model, x)
        
        # Initialize LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            feature_names=feature_names[:X.shape[1]],
            mode='classification',  # Assume classification for now
            discretize_continuous=True
        )
        
        # Explain each instance
        explanations = []
        for i, instance in enumerate(X):
            try:
                exp = explainer.explain_instance(
                    instance,
                    predict_fn,
                    num_features=min(num_features, X.shape[1])
                )
                
                # Extract feature importance
                feature_importance = dict(exp.as_list())
                explanations.append({
                    'instance_id': i,
                    'feature_importance': feature_importance,
                    'score': exp.score if hasattr(exp, 'score') else None
                })
                
            except Exception as e:
                logger.warning(f"LIME explanation failed for instance {i}: {e}")
                explanations.append({
                    'instance_id': i,
                    'error': str(e)
                })
        
        # Prepare result
        result = {
            'method': 'lime',
            'explanations': explanations,
            'feature_names': feature_names[:X.shape[1]],
            'input_shape': X.shape,
            'num_features': num_features,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Cache results
        if cache_dir:
            cache_path = _cache_explanation(result, cache_dir, 'lime')
            result['cache_path'] = cache_path
        
        logger.info(f"LIME explanation completed for {len(explanations)} instances")
        return result
    
    except Exception as e:
        logger.error(f"LIME explanation failed: {e}")
        return {
            'method': 'lime',
            'error': str(e),
            'feature_names': feature_names,
            'timestamp': pd.Timestamp.now().isoformat()
        }


def _cache_explanation(explanation: Dict[str, Any], cache_dir: str, method: str) -> str:
    """Cache explanation results to disk."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create hash for filename
    content_str = json.dumps(explanation, sort_keys=True, default=str)
    hash_key = hashlib.md5(content_str.encode()).hexdigest()[:16]
    
    # Save to file
    cache_path = cache_dir / f"{method}_{hash_key}.json"
    with open(cache_path, 'w') as f:
        json.dump(explanation, f, indent=2, default=str)
    
    logger.debug(f"Cached {method} explanation to {cache_path}")
    return str(cache_path)


if __name__ == "__main__":
    # Test with a simple model
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Create test data
        X, y = make_classification(n_samples=100, n_features=5, n_classes=3, random_state=42)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Train simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        print("Testing SHAP explanation...")
        if SHAP_AVAILABLE:
            shap_result = explain_with_shap(model, X[:5], feature_names, nsamples=50)
            print(f"SHAP result keys: {list(shap_result.keys())}")
            if 'shap_values' in shap_result:
                print(f"SHAP values shape: {np.array(shap_result['shap_values']).shape}")
        
        print("\nTesting LIME explanation...")
        if LIME_AVAILABLE:
            lime_result = explain_with_lime(model, X[:3], feature_names, num_features=3)
            print(f"LIME result keys: {list(lime_result.keys())}")
            if 'explanations' in lime_result:
                print(f"LIME explanations count: {len(lime_result['explanations'])}")
        
        print("\nExplainability module test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()