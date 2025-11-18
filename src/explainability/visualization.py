"""Visualization utilities for explainability results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import json

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def plot_shap_summary(
    explanation: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Dict[str, Any]:
    """
    Create SHAP summary plot.
    
    Args:
        explanation: SHAP explanation results
        save_path: Optional path to save plot
        figsize: Figure size
    
    Returns:
        Plot metadata
    """
    if 'shap_values' not in explanation:
        raise ValueError("No SHAP values found in explanation")
    
    shap_values = np.array(explanation['shap_values'])
    feature_names = explanation.get('feature_names', [])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Feature importance (mean absolute SHAP values)
    if shap_values.ndim == 2:
        importance = np.mean(np.abs(shap_values), axis=0)
        
        # Sort features by importance
        sorted_idx = np.argsort(importance)[-10:]  # Top 10
        sorted_importance = importance[sorted_idx]
        sorted_names = [feature_names[i] if i < len(feature_names) else f'feature_{i}' 
                       for i in sorted_idx]
        
        ax1.barh(range(len(sorted_importance)), sorted_importance)
        ax1.set_yticks(range(len(sorted_importance)))
        ax1.set_yticklabels(sorted_names)
        ax1.set_xlabel('Mean |SHAP value|')
        ax1.set_title('Feature Importance')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: SHAP values distribution
    if shap_values.ndim == 2 and shap_values.shape[0] > 1:
        # Box plot of SHAP values for top features
        top_features = sorted_idx[-5:]  # Top 5
        shap_subset = shap_values[:, top_features]
        
        ax2.boxplot(shap_subset, labels=[
            feature_names[i] if i < len(feature_names) else f'feature_{i}'
            for i in top_features
        ])
        ax2.set_ylabel('SHAP value')
        ax2.set_title('SHAP Values Distribution')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved SHAP summary plot to {save_path}")
    
    # Prepare metadata
    metadata = {
        'plot_type': 'shap_summary',
        'n_features': len(feature_names),
        'n_samples': shap_values.shape[0] if shap_values.ndim > 1 else 1,
        'top_features': sorted_names[-5:] if 'sorted_names' in locals() else []
    }
    
    if save_path:
        metadata['save_path'] = save_path
    
    return metadata


def plot_lime_explanation(
    explanation: Dict[str, Any],
    instance_idx: int = 0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Dict[str, Any]:
    """
    Create LIME explanation plot for a specific instance.
    
    Args:
        explanation: LIME explanation results
        instance_idx: Index of instance to plot
        save_path: Optional path to save plot
        figsize: Figure size
    
    Returns:
        Plot metadata
    """
    if 'explanations' not in explanation:
        raise ValueError("No LIME explanations found")
    
    explanations = explanation['explanations']
    if instance_idx >= len(explanations):
        raise ValueError(f"Instance index {instance_idx} out of range")
    
    instance_exp = explanations[instance_idx]
    if 'feature_importance' not in instance_exp:
        raise ValueError(f"No feature importance found for instance {instance_idx}")
    
    feature_importance = instance_exp['feature_importance']
    
    # Sort by absolute importance
    sorted_items = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    features, importances = zip(*sorted_items[:15])  # Top 15
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color bars by positive/negative
    colors = ['green' if imp > 0 else 'red' for imp in importances]
    
    bars = ax.barh(range(len(features)), importances, color=colors, alpha=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'LIME Explanation - Instance {instance_idx}')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, imp) in enumerate(zip(bars, importances)):
        ax.text(imp + (0.01 if imp > 0 else -0.01), i, f'{imp:.3f}', 
                va='center', ha='left' if imp > 0 else 'right')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved LIME explanation plot to {save_path}")
    
    metadata = {
        'plot_type': 'lime_explanation',
        'instance_idx': instance_idx,
        'n_features_shown': len(features),
        'top_positive': max(importances) if importances else 0,
        'top_negative': min(importances) if importances else 0
    }
    
    if save_path:
        metadata['save_path'] = save_path
    
    return metadata


def plot_attention_heatmap(
    explanation: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> Dict[str, Any]:
    """
    Create attention heatmap visualization.
    
    Args:
        explanation: Attention explanation results
        save_path: Optional path to save plot
        figsize: Figure size
    
    Returns:
        Plot metadata
    """
    if 'attention_weights' not in explanation:
        raise ValueError("No attention weights found in explanation")
    
    attention_weights = np.array(explanation['attention_weights'])
    feature_names = explanation.get('feature_names', [])
    
    # Ensure 2D
    if attention_weights.ndim != 2:
        if attention_weights.ndim == 1:
            attention_weights = attention_weights.reshape(1, -1)
        else:
            attention_weights = attention_weights.reshape(attention_weights.shape[0], -1)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use feature names if available
    x_labels = feature_names[:attention_weights.shape[1]] if feature_names else None
    y_labels = feature_names[:attention_weights.shape[0]] if feature_names else None
    
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    
    # Set ticks and labels
    if x_labels:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    if y_labels:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight')
    
    ax.set_title('Attention Weights Heatmap')
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved attention heatmap to {save_path}")
    
    metadata = {
        'plot_type': 'attention_heatmap',
        'attention_shape': attention_weights.shape,
        'max_attention': float(np.max(attention_weights)),
        'min_attention': float(np.min(attention_weights)),
        'mean_attention': float(np.mean(attention_weights))
    }
    
    if save_path:
        metadata['save_path'] = save_path
    
    return metadata


def create_explanation_dashboard(
    explanations: List[Dict[str, Any]],
    save_dir: str = "artifacts/explanations/dashboard",
    title: str = "Model Explanations Dashboard"
) -> Dict[str, Any]:
    """
    Create a comprehensive dashboard with multiple explanation visualizations.
    
    Args:
        explanations: List of explanation results from different methods
        save_dir: Directory to save dashboard files
        title: Dashboard title
    
    Returns:
        Dashboard metadata
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    dashboard_data = {
        'title': title,
        'explanations': [],
        'plots': [],
        'summary': {}
    }
    
    # Process each explanation
    for i, explanation in enumerate(explanations):
        method = explanation.get('method', 'unknown')
        
        try:
            if method == 'shap' and 'shap_values' in explanation:
                plot_path = save_dir / f"shap_summary_{i}.png"
                plot_meta = plot_shap_summary(explanation, str(plot_path))
                dashboard_data['plots'].append(plot_meta)
            
            elif method == 'lime' and 'explanations' in explanation:
                plot_path = save_dir / f"lime_explanation_{i}.png"
                plot_meta = plot_lime_explanation(explanation, save_path=str(plot_path))
                dashboard_data['plots'].append(plot_meta)
            
            elif method == 'attention' and 'attention_weights' in explanation:
                plot_path = save_dir / f"attention_heatmap_{i}.png"
                plot_meta = plot_attention_heatmap(explanation, save_path=str(plot_path))
                dashboard_data['plots'].append(plot_meta)
            
            # Add explanation metadata
            dashboard_data['explanations'].append({
                'method': method,
                'index': i,
                'has_error': 'error' in explanation,
                'input_shape': explanation.get('input_shape'),
                'feature_count': len(explanation.get('feature_names', []))
            })
            
        except Exception as e:
            logger.error(f"Failed to create plot for explanation {i} ({method}): {e}")
            dashboard_data['explanations'].append({
                'method': method,
                'index': i,
                'has_error': True,
                'error': str(e)
            })
    
    # Create summary statistics
    methods_used = [exp['method'] for exp in dashboard_data['explanations']]
    dashboard_data['summary'] = {
        'total_explanations': len(explanations),
        'methods_used': list(set(methods_used)),
        'successful_plots': len(dashboard_data['plots']),
        'failed_explanations': sum(1 for exp in dashboard_data['explanations'] if exp.get('has_error', False))
    }
    
    # Save dashboard metadata
    metadata_path = save_dir / "dashboard_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2, default=str)
    
    logger.info(f"Created explanation dashboard in {save_dir}")
    logger.info(f"Generated {len(dashboard_data['plots'])} plots for {len(explanations)} explanations")
    
    return dashboard_data


def plot_feature_importance_comparison(
    explanations: List[Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> Dict[str, Any]:
    """
    Compare feature importance across different explanation methods.
    
    Args:
        explanations: List of explanation results
        save_path: Optional path to save plot
        figsize: Figure size
    
    Returns:
        Comparison metadata
    """
    # Extract feature importance from different methods
    importance_data = {}
    all_features = set()
    
    for explanation in explanations:
        method = explanation.get('method', 'unknown')
        feature_names = explanation.get('feature_names', [])
        
        if method == 'shap' and 'shap_values' in explanation:
            shap_values = np.array(explanation['shap_values'])
            if shap_values.ndim == 2:
                importance = np.mean(np.abs(shap_values), axis=0)
                importance_data['SHAP'] = dict(zip(feature_names, importance))
                all_features.update(feature_names)
        
        elif method == 'lime' and 'explanations' in explanation:
            # Average LIME importance across instances
            lime_importance = {}
            for instance_exp in explanation['explanations']:
                if 'feature_importance' in instance_exp:
                    for feature, imp in instance_exp['feature_importance'].items():
                        if feature not in lime_importance:
                            lime_importance[feature] = []
                        lime_importance[feature].append(abs(imp))
            
            # Average across instances
            avg_lime_importance = {
                feature: np.mean(values) 
                for feature, values in lime_importance.items()
            }
            importance_data['LIME'] = avg_lime_importance
            all_features.update(avg_lime_importance.keys())
    
    if not importance_data:
        raise ValueError("No valid importance data found in explanations")
    
    # Create comparison plot
    all_features = sorted(list(all_features))[:15]  # Top 15 features
    methods = list(importance_data.keys())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(all_features))
    width = 0.35 if len(methods) == 2 else 0.25
    
    for i, method in enumerate(methods):
        values = [importance_data[method].get(feature, 0) for feature in all_features]
        ax.bar(x + i * width, values, width, label=method, alpha=0.8)
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance Comparison')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(all_features, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved feature importance comparison to {save_path}")
    
    metadata = {
        'plot_type': 'feature_importance_comparison',
        'methods_compared': methods,
        'n_features': len(all_features),
        'top_features': all_features[:5]
    }
    
    if save_path:
        metadata['save_path'] = save_path
    
    return metadata


if __name__ == "__main__":
    # Test visualization functions
    try:
        print("Testing visualization utilities...")
        
        # Create dummy explanation data
        dummy_shap = {
            'method': 'shap',
            'shap_values': np.random.randn(50, 10).tolist(),
            'feature_names': [f'feature_{i}' for i in range(10)]
        }
        
        dummy_lime = {
            'method': 'lime',
            'explanations': [
                {
                    'instance_id': i,
                    'feature_importance': {
                        f'feature_{j}': np.random.randn() 
                        for j in range(10)
                    }
                }
                for i in range(5)
            ],
            'feature_names': [f'feature_{i}' for i in range(10)]
        }
        
        dummy_attention = {
            'method': 'attention',
            'attention_weights': np.random.random((10, 10)).tolist(),
            'feature_names': [f'feature_{i}' for i in range(10)]
        }
        
        # Test individual plots
        print("Testing SHAP summary plot...")
        shap_meta = plot_shap_summary(dummy_shap)
        print(f"SHAP plot metadata: {shap_meta}")
        
        print("Testing LIME explanation plot...")
        lime_meta = plot_lime_explanation(dummy_lime)
        print(f"LIME plot metadata: {lime_meta}")
        
        print("Testing attention heatmap...")
        attention_meta = plot_attention_heatmap(dummy_attention)
        print(f"Attention plot metadata: {attention_meta}")
        
        # Test dashboard creation
        print("Testing dashboard creation...")
        dashboard_meta = create_explanation_dashboard(
            [dummy_shap, dummy_lime, dummy_attention],
            save_dir="test_dashboard"
        )
        print(f"Dashboard metadata: {dashboard_meta['summary']}")
        
        # Test comparison plot
        print("Testing feature importance comparison...")
        comparison_meta = plot_feature_importance_comparison([dummy_shap, dummy_lime])
        print(f"Comparison metadata: {comparison_meta}")
        
        print("Visualization utilities test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()