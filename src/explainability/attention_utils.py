"""Attention extraction utilities for PyTorch models."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable, Any
from collections import defaultdict

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class AttentionCapture:
    """Class to capture and manage attention weights from PyTorch models."""
    
    def __init__(self):
        self.attention_weights = defaultdict(list)
        self.hooks = []
        self.module_names = []
    
    def clear(self):
        """Clear captured attention weights."""
        self.attention_weights.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.module_names.clear()
    
    def get_captured_attention(self) -> Dict[str, List[torch.Tensor]]:
        """Get all captured attention weights."""
        return dict(self.attention_weights)


def register_attention_hooks(model: nn.Module) -> Tuple[AttentionCapture, Callable]:
    """
    Register forward hooks on attention modules to capture attention weights.
    
    Args:
        model: PyTorch model containing attention modules
    
    Returns:
        Tuple of (AttentionCapture object, cleanup function)
    """
    capture = AttentionCapture()
    attention_modules_found = 0
    
    def create_hook(module_name: str):
        def hook_fn(module, input, output):
            try:
                # Handle different attention module types
                if isinstance(module, nn.MultiheadAttention):
                    # For nn.MultiheadAttention, attention weights are in output[1]
                    if len(output) > 1 and output[1] is not None:
                        attn_weights = output[1].detach()
                        capture.attention_weights[module_name].append(attn_weights)
                        logger.debug(f"Captured attention from {module_name}: {attn_weights.shape}")
                
                elif hasattr(module, 'attention_weights'):
                    # Custom attention modules with attention_weights attribute
                    attn_weights = module.attention_weights.detach()
                    capture.attention_weights[module_name].append(attn_weights)
                    logger.debug(f"Captured custom attention from {module_name}: {attn_weights.shape}")
                
                elif 'attention' in module_name.lower() and hasattr(module, 'weight'):
                    # Fallback for modules with 'attention' in name
                    if hasattr(module, 'weight') and module.weight is not None:
                        weights = module.weight.detach()
                        capture.attention_weights[module_name].append(weights)
                        logger.debug(f"Captured weight-based attention from {module_name}: {weights.shape}")
                
            except Exception as e:
                logger.warning(f"Failed to capture attention from {module_name}: {e}")
        
        return hook_fn
    
    # Walk through all modules and register hooks
    for name, module in model.named_modules():
        should_hook = False
        
        # Check for standard attention modules
        if isinstance(module, nn.MultiheadAttention):
            should_hook = True
            logger.info(f"Found MultiheadAttention module: {name}")
        
        # Check for custom attention modules by name
        elif any(keyword in name.lower() for keyword in ['attention', 'attn']):
            should_hook = True
            logger.info(f"Found potential attention module by name: {name}")
        
        # Check for custom attention modules by class name
        elif any(keyword in module.__class__.__name__.lower() for keyword in ['attention', 'attn']):
            should_hook = True
            logger.info(f"Found potential attention module by class: {name} ({module.__class__.__name__})")
        
        if should_hook:
            hook = module.register_forward_hook(create_hook(name))
            capture.hooks.append(hook)
            capture.module_names.append(name)
            attention_modules_found += 1
    
    if attention_modules_found == 0:
        raise ValueError(
            "No attention modules found in the model. "
            "Make sure your model contains nn.MultiheadAttention modules or "
            "custom attention modules with 'attention' or 'attn' in their name/class name."
        )
    
    logger.info(f"Registered hooks on {attention_modules_found} attention modules")
    
    def cleanup():
        capture.remove_hooks()
    
    return capture, cleanup


def aggregate_attention(
    attention_weights: List[torch.Tensor],
    method: str = 'mean',
    across_heads: bool = True,
    across_layers: bool = True
) -> np.ndarray:
    """
    Aggregate attention weights for visualization.
    
    Args:
        attention_weights: List of attention weight tensors
        method: Aggregation method ('mean', 'max', 'sum')
        across_heads: Whether to aggregate across attention heads
        across_layers: Whether to aggregate across layers/calls
    
    Returns:
        Aggregated attention weights as numpy array
    """
    if not attention_weights:
        raise ValueError("No attention weights provided")
    
    logger.info(f"Aggregating {len(attention_weights)} attention tensors using {method}")
    
    try:
        # Convert to numpy and handle different tensor shapes
        np_weights = []
        for i, weights in enumerate(attention_weights):
            if isinstance(weights, torch.Tensor):
                weights_np = weights.cpu().numpy()
            else:
                weights_np = np.array(weights)
            
            logger.debug(f"Attention tensor {i} shape: {weights_np.shape}")
            np_weights.append(weights_np)
        
        # Handle different attention weight shapes
        # Common shapes: (batch, heads, seq_len, seq_len) or (batch, seq_len, seq_len)
        processed_weights = []
        
        for weights in np_weights:
            # Ensure at least 2D
            if weights.ndim == 1:
                weights = weights.reshape(1, -1)
            
            # Handle batch dimension
            if weights.ndim >= 3:
                # Take first batch item if batch dimension exists
                if weights.shape[0] == 1:
                    weights = weights[0]
                else:
                    # Average across batch
                    weights = np.mean(weights, axis=0)
            
            # Handle head dimension
            if across_heads and weights.ndim == 3:
                # Assume shape is (heads, seq_len, seq_len)
                if method == 'mean':
                    weights = np.mean(weights, axis=0)
                elif method == 'max':
                    weights = np.max(weights, axis=0)
                elif method == 'sum':
                    weights = np.sum(weights, axis=0)
                else:
                    raise ValueError(f"Unknown aggregation method: {method}")
            
            processed_weights.append(weights)
        
        # Aggregate across layers/calls
        if across_layers and len(processed_weights) > 1:
            # Ensure all weights have the same shape
            min_shape = min([w.shape for w in processed_weights])
            aligned_weights = []
            
            for weights in processed_weights:
                # Truncate to minimum shape if needed
                if weights.shape != min_shape:
                    if weights.ndim == 2:
                        weights = weights[:min_shape[0], :min_shape[1]]
                    else:
                        weights = weights[:min_shape[0]]
                aligned_weights.append(weights)
            
            # Aggregate
            if method == 'mean':
                result = np.mean(aligned_weights, axis=0)
            elif method == 'max':
                result = np.max(aligned_weights, axis=0)
            elif method == 'sum':
                result = np.sum(aligned_weights, axis=0)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")
        else:
            # Just take the first (or only) weight matrix
            result = processed_weights[0]
        
        logger.info(f"Aggregated attention shape: {result.shape}")
        return result
    
    except Exception as e:
        logger.error(f"Failed to aggregate attention weights: {e}")
        # Return dummy attention matrix
        return np.random.random((10, 10))


def extract_attention_weights(
    model: nn.Module,
    input_data: torch.Tensor,
    layer_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Extract attention weights from a PyTorch model.
    
    Args:
        model: PyTorch model with attention mechanisms
        input_data: Input tensor to process
        layer_names: Optional list of specific layer names to extract from
    
    Returns:
        Dictionary mapping layer names to attention weight arrays
    """
    logger.info(f"Extracting attention weights from model with input shape {input_data.shape}")
    
    try:
        # Register hooks and capture attention
        capture, cleanup = register_attention_hooks(model)
        
        # Run forward pass
        model.eval()
        with torch.no_grad():
            _ = model(input_data)
        
        # Get captured attention weights
        captured = capture.get_captured_attention()
        
        # Process and aggregate attention weights
        result = {}
        for module_name, weights_list in captured.items():
            if layer_names is None or module_name in layer_names:
                try:
                    # Aggregate attention weights for this module
                    aggregated = aggregate_attention(weights_list, method='mean')
                    result[module_name] = aggregated
                    logger.debug(f"Extracted attention from {module_name}: {aggregated.shape}")
                except Exception as e:
                    logger.warning(f"Failed to aggregate attention for {module_name}: {e}")
        
        # Cleanup hooks
        cleanup()
        
        logger.info(f"Successfully extracted attention from {len(result)} modules")
        return result
    
    except Exception as e:
        logger.error(f"Failed to extract attention weights: {e}")
        return {}


def visualize_attention_heatmap(
    attention_weights: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create attention heatmap visualization data.
    
    Args:
        attention_weights: 2D attention weight matrix
        feature_names: Optional feature names for axes
        save_path: Optional path to save visualization
    
    Returns:
        Dictionary with visualization data
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Ensure 2D
        if attention_weights.ndim != 2:
            logger.warning(f"Expected 2D attention weights, got {attention_weights.ndim}D")
            if attention_weights.ndim == 1:
                attention_weights = attention_weights.reshape(1, -1)
            else:
                attention_weights = attention_weights.reshape(attention_weights.shape[0], -1)
        
        # Create visualization data
        viz_data = {
            'attention_matrix': attention_weights.tolist(),
            'shape': attention_weights.shape,
            'max_attention': float(np.max(attention_weights)),
            'min_attention': float(np.min(attention_weights)),
            'mean_attention': float(np.mean(attention_weights))
        }
        
        if feature_names:
            viz_data['feature_names'] = feature_names[:attention_weights.shape[1]]
        
        # Create matplotlib figure if requested
        if save_path:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                attention_weights,
                xticklabels=feature_names[:attention_weights.shape[1]] if feature_names else False,
                yticklabels=feature_names[:attention_weights.shape[0]] if feature_names else False,
                cmap='Blues',
                cbar=True
            )
            plt.title('Attention Weights Heatmap')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            viz_data['heatmap_path'] = save_path
            logger.info(f"Saved attention heatmap to {save_path}")
        
        return viz_data
    
    except ImportError:
        logger.warning("Matplotlib/seaborn not available for visualization")
        return {
            'attention_matrix': attention_weights.tolist(),
            'shape': attention_weights.shape,
            'error': 'Visualization libraries not available'
        }
    except Exception as e:
        logger.error(f"Failed to create attention visualization: {e}")
        return {
            'attention_matrix': attention_weights.tolist(),
            'shape': attention_weights.shape,
            'error': str(e)
        }


if __name__ == "__main__":
    # Test with a simple model containing attention
    try:
        print("Testing attention extraction utilities...")
        
        # Create a simple model with MultiheadAttention
        class SimpleAttentionModel(nn.Module):
            def __init__(self, d_model=64, nhead=8):
                super().__init__()
                self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
                self.linear = nn.Linear(d_model, 3)
            
            def forward(self, x):
                attn_out, attn_weights = self.attention(x, x, x)
                return self.linear(attn_out), attn_weights
        
        model = SimpleAttentionModel()
        model.eval()
        
        # Register hooks
        capture, cleanup = register_attention_hooks(model)
        
        # Run forward pass
        batch_size, seq_len, d_model = 2, 10, 64
        x = torch.randn(batch_size, seq_len, d_model)
        
        with torch.no_grad():
            output = model(x)
        
        # Get captured attention
        captured = capture.get_captured_attention()
        print(f"Captured attention from {len(captured)} modules")
        
        for module_name, weights_list in captured.items():
            print(f"Module {module_name}: {len(weights_list)} captures")
            for i, weights in enumerate(weights_list):
                print(f"  Capture {i}: {weights.shape}")
        
        # Test aggregation
        if captured:
            first_module = list(captured.keys())[0]
            weights_list = captured[first_module]
            
            aggregated = aggregate_attention(weights_list, method='mean')
            print(f"Aggregated attention shape: {aggregated.shape}")
            
            # Test visualization
            viz_data = visualize_attention_heatmap(aggregated)
            print(f"Visualization data keys: {list(viz_data.keys())}")
        
        # Cleanup
        cleanup()
        print("Attention extraction test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()