"""Unified explainability interface for all explanation methods."""

import numpy as np
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import warnings

try:
    from src.explainability.shap_lime import explain_with_shap, explain_with_lime
    from src.explainability.attention_utils import register_attention_hooks, aggregate_attention
    from src.explainability.cache import make_input_hash, save_explanation, load_explanation
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def preprocess_observation(
    observation: Union[np.ndarray, Dict[str, Any]],
    method: str = 'shap'
) -> np.ndarray:
    """
    Preprocess observation for explanation methods.
    
    Args:
        observation: Input observation (can be sequence or flattened)
        method: Explanation method ('shap', 'lime', 'attention')
    
    Returns:
        Preprocessed observation as numpy array
    """
    if isinstance(observation, dict):
        # Extract main observation if wrapped in dict
        if 'observation' in observation:
            obs = observation['observation']
        else:
            # Take first array value
            obs = next(iter(observation.values()))
    else:
        obs = observation
    
    # Convert to numpy array
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    
    # Handle different methods
    if method == 'attention':
        # Keep sequence structure for attention
        if obs.ndim == 1:
            # Assume flattened sequence, try to reshape
            seq_len = int(np.sqrt(len(obs)))
            if seq_len * seq_len == len(obs):
                obs = obs.reshape(seq_len, seq_len)
            else:
                # Default reshape
                obs = obs.reshape(1, -1)
        elif obs.ndim == 3:
            # Remove batch dimension if present
            if obs.shape[0] == 1:
                obs = obs[0]
    else:
        # Flatten for SHAP/LIME
        if obs.ndim > 2:
            obs = obs.reshape(obs.shape[0], -1)
        elif obs.ndim == 1:
            obs = obs.reshape(1, -1)
    
    return obs.astype(np.float32)


def generate_feature_names(observation: np.ndarray, prefix: str = 'feature') -> List[str]:
    """Generate feature names for observation."""
    if observation.ndim == 1:
        n_features = len(observation)
    else:
        n_features = observation.shape[-1]
    
    return [f'{prefix}_{i}' for i in range(n_features)]


def explain_instance(
    model,
    observation: Union[np.ndarray, Dict[str, Any]],
    method: str = 'shap',
    feature_names: Optional[List[str]] = None,
    cache: bool = True,
    cache_dir: str = "artifacts/explanations",
    **kwargs
) -> Dict[str, Any]:
    """
    Unified interface for explaining model predictions.
    
    Args:
        model: Trained model to explain
        observation: Input observation to explain
        method: Explanation method ('shap', 'lime', 'attention', 'rule')
        feature_names: Optional feature names
        cache: Whether to cache results
        cache_dir: Directory for caching
        **kwargs: Additional arguments for specific methods
    
    Returns:
        Dictionary with explanation results and metadata
    """
    logger.info(f"Explaining instance using method: {method}")
    
    try:
        # Preprocess observation
        X = preprocess_observation(observation, method)
        
        # Generate feature names if not provided
        if feature_names is None:
            feature_names = generate_feature_names(X)
        
        # Generate model ID for caching
        model_id = _get_model_id(model)
        
        # Check cache first
        if cache:
            input_hash = make_input_hash(model_id, method, X)
            cached_result = load_explanation(input_hash, cache_dir)
            if cached_result is not None:
                logger.info("Loaded explanation from cache")
                cached_result['from_cache'] = True
                return cached_result
        
        # Generate explanation based on method
        if method == 'shap':
            result = explain_with_shap(
                model, X, feature_names,
                nsamples=kwargs.get('nsamples', 100),
                cache_dir=cache_dir if cache else None
            )
        
        elif method == 'lime':
            result = explain_with_lime(
                model, X, feature_names,
                num_features=kwargs.get('num_features', 10),
                cache_dir=cache_dir if cache else None
            )
        
        elif method == 'attention':
            result = _explain_with_attention(
                model, X, feature_names, **kwargs
            )
        
        elif method == 'rule':
            result = _explain_with_rules(
                model, observation, feature_names, **kwargs
            )
        
        else:
            raise ValueError(f"Unknown explanation method: {method}")
        
        # Add metadata
        result.update({
            'model_id': model_id,
            'input_hash': make_input_hash(model_id, method, X) if cache else None,
            'input_shape': X.shape,
            'from_cache': False
        })
        
        # Cache result
        if cache and 'error' not in result:
            save_explanation(result, cache_dir)
        
        logger.info(f"Explanation completed using {method}")
        return result
    
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        return {
            'method': method,
            'error': str(e),
            'model_id': _get_model_id(model),
            'input_shape': getattr(observation, 'shape', None),
            'feature_names': feature_names or []
        }


def _explain_with_attention(
    model,
    X: np.ndarray,
    feature_names: List[str],
    **kwargs
) -> Dict[str, Any]:
    """Explain using attention weights."""
    try:
        import torch
        
        # Register attention hooks
        capture, cleanup = register_attention_hooks(model)
        
        # Run forward pass
        model.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X_tensor = torch.FloatTensor(X)
                if X_tensor.ndim == 2:
                    X_tensor = X_tensor.unsqueeze(0)  # Add batch dimension
            else:
                X_tensor = X
            
            # Forward pass to capture attention
            _ = model(X_tensor)
        
        # Get captured attention
        captured = capture.get_captured_attention()
        
        if not captured:
            cleanup()
            return {
                'method': 'attention',
                'error': 'No attention weights captured',
                'feature_names': feature_names
            }
        
        # Aggregate attention from all modules
        all_attention = []
        attention_info = {}
        
        for module_name, weights_list in captured.items():
            if weights_list:
                aggregated = aggregate_attention(
                    weights_list,
                    method=kwargs.get('aggregate_method', 'mean')
                )
                all_attention.append(aggregated)
                attention_info[module_name] = {
                    'shape': aggregated.shape,
                    'num_captures': len(weights_list)
                }
        
        # Cleanup hooks
        cleanup()
        
        if not all_attention:
            return {
                'method': 'attention',
                'error': 'No valid attention weights found',
                'feature_names': feature_names
            }
        
        # Use first attention matrix or average if multiple
        if len(all_attention) == 1:
            final_attention = all_attention[0]
        else:
            # Average attention across modules
            min_shape = min([att.shape for att in all_attention])
            aligned_attention = []
            for att in all_attention:
                if att.shape != min_shape:
                    att = att[:min_shape[0], :min_shape[1]]
                aligned_attention.append(att)
            final_attention = np.mean(aligned_attention, axis=0)
        
        return {
            'method': 'attention',
            'attention_weights': final_attention.tolist(),
            'attention_shape': final_attention.shape,
            'feature_names': feature_names[:final_attention.shape[1]],
            'modules_info': attention_info,
            'aggregate_method': kwargs.get('aggregate_method', 'mean')
        }
    
    except Exception as e:
        return {
            'method': 'attention',
            'error': str(e),
            'feature_names': feature_names
        }


def _explain_with_rules(
    model,
    observation: Union[np.ndarray, Dict[str, Any]],
    feature_names: List[str],
    **kwargs
) -> Dict[str, Any]:
    """Explain using rule-based logic."""
    try:
        # This is a placeholder for rule-based explanations
        # In practice, this would integrate with the rule system
        
        rules_triggered = []
        rule_explanations = {}
        
        # Example rule checks (placeholder)
        if isinstance(observation, np.ndarray):
            obs_dict = {f'feature_{i}': val for i, val in enumerate(observation.flatten())}
        else:
            obs_dict = observation
        
        # Simulate rule checking
        for i, (key, value) in enumerate(obs_dict.items()):
            if isinstance(value, (int, float)):
                if abs(value) > 2.0:  # Example threshold
                    rules_triggered.append(f'high_value_rule_{i}')
                    rule_explanations[f'high_value_rule_{i}'] = {
                        'feature': key,
                        'value': float(value),
                        'threshold': 2.0,
                        'description': f'Feature {key} exceeds threshold'
                    }
        
        return {
            'method': 'rule',
            'rules_triggered': rules_triggered,
            'rule_explanations': rule_explanations,
            'feature_names': feature_names,
            'total_rules_checked': len(obs_dict)
        }
    
    except Exception as e:
        return {
            'method': 'rule',
            'error': str(e),
            'feature_names': feature_names
        }


def _get_model_id(model) -> str:
    """Generate a unique ID for the model."""
    try:
        # Try to get model class name and parameters
        model_info = f"{model.__class__.__name__}"
        
        # Add parameter count if available
        if hasattr(model, 'parameters'):
            try:
                param_count = sum(p.numel() for p in model.parameters())
                model_info += f"_params_{param_count}"
            except:
                pass
        
        # Add hash of model state if available
        if hasattr(model, 'state_dict'):
            try:
                import torch
                state_str = str(list(model.state_dict().keys()))
                state_hash = hashlib.md5(state_str.encode()).hexdigest()[:8]
                model_info += f"_state_{state_hash}"
            except:
                pass
        
        return model_info
    
    except:
        return "unknown_model"


def pretty_print_explanation(explanation: Dict[str, Any]) -> str:
    """Pretty print explanation results for CLI."""
    method = explanation.get('method', 'unknown')
    
    if 'error' in explanation:
        return f"❌ {method.upper()} explanation failed: {explanation['error']}"
    
    lines = [f"📊 {method.upper()} Explanation Results"]
    lines.append("=" * 40)
    
    if method == 'shap':
        if 'shap_values' in explanation:
            shap_values = np.array(explanation['shap_values'])
            lines.append(f"SHAP values shape: {shap_values.shape}")
            lines.append(f"Base value: {explanation.get('base_values', 'N/A')}")
            
            # Show top features
            if len(shap_values.shape) == 2 and shap_values.shape[0] > 0:
                feature_names = explanation.get('feature_names', [])
                avg_importance = np.mean(np.abs(shap_values), axis=0)
                top_indices = np.argsort(avg_importance)[-5:][::-1]
                
                lines.append("\nTop 5 most important features:")
                for i, idx in enumerate(top_indices):
                    if idx < len(feature_names):
                        name = feature_names[idx]
                    else:
                        name = f"feature_{idx}"
                    lines.append(f"  {i+1}. {name}: {avg_importance[idx]:.4f}")
    
    elif method == 'lime':
        explanations = explanation.get('explanations', [])
        lines.append(f"LIME explanations for {len(explanations)} instances")
        
        if explanations and 'feature_importance' in explanations[0]:
            # Show first instance
            first_exp = explanations[0]['feature_importance']
            lines.append("\nTop features for first instance:")
            sorted_features = sorted(first_exp.items(), key=lambda x: abs(x[1]), reverse=True)
            for i, (feature, importance) in enumerate(sorted_features[:5]):
                lines.append(f"  {i+1}. {feature}: {importance:.4f}")
    
    elif method == 'attention':
        if 'attention_weights' in explanation:
            shape = explanation.get('attention_shape', 'unknown')
            lines.append(f"Attention weights shape: {shape}")
            
            modules_info = explanation.get('modules_info', {})
            lines.append(f"Attention modules: {len(modules_info)}")
            for module, info in modules_info.items():
                lines.append(f"  - {module}: {info.get('shape', 'unknown')}")
    
    elif method == 'rule':
        rules_triggered = explanation.get('rules_triggered', [])
        lines.append(f"Rules triggered: {len(rules_triggered)}")
        for rule in rules_triggered:
            lines.append(f"  - {rule}")
    
    # Add metadata
    if 'from_cache' in explanation:
        cache_status = "from cache" if explanation['from_cache'] else "computed"
        lines.append(f"\nResult: {cache_status}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the unified interface
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        print("Testing unified explainability interface...")
        
        # Create test data and model
        X, y = make_classification(n_samples=100, n_features=8, n_classes=3, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test observation
        test_obs = X[0]
        feature_names = [f'feature_{i}' for i in range(len(test_obs))]
        
        # Test different explanation methods
        methods = ['shap', 'lime', 'rule']
        
        for method in methods:
            print(f"\n--- Testing {method.upper()} ---")
            try:
                result = explain_instance(
                    model, test_obs, method=method,
                    feature_names=feature_names,
                    cache=False
                )
                
                print(pretty_print_explanation(result))
                
            except Exception as e:
                print(f"❌ {method} failed: {e}")
        
        print("\nUnified interface test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()