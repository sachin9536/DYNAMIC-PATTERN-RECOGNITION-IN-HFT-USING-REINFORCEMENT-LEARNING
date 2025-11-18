"""Explainability module for market anomaly detection."""

# Import main interface
try:
    from .interface import explain_instance, pretty_print_explanation
    from .shap_lime import explain_with_shap, explain_with_lime
    from .attention_utils import register_attention_hooks, aggregate_attention
    from .cache import make_input_hash, save_explanation, load_explanation
    from .rule_based import MarketAnomalyRules, Rule, RuleType
    from .visualization import (
        plot_shap_summary, plot_lime_explanation, plot_attention_heatmap,
        create_explanation_dashboard, plot_feature_importance_comparison
    )
    
    __all__ = [
        # Core interface
        'explain_instance',
        'pretty_print_explanation',
        
        # Method-specific functions
        'explain_with_shap',
        'explain_with_lime',
        'register_attention_hooks',
        'aggregate_attention',
        
        # Rule-based system
        'MarketAnomalyRules',
        'Rule',
        'RuleType',
        
        # Visualization
        'plot_shap_summary',
        'plot_lime_explanation', 
        'plot_attention_heatmap',
        'create_explanation_dashboard',
        'plot_feature_importance_comparison',
        
        # Caching
        'make_input_hash',
        'save_explanation',
        'load_explanation'
    ]
    
except ImportError as e:
    print(f"Warning: Some explainability features may not be available: {e}")
    __all__ = []