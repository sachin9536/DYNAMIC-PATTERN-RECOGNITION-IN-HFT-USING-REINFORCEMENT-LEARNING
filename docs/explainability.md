# Explainability Module Documentation

The explainability module provides comprehensive tools for understanding and interpreting AI model decisions in market anomaly detection. It supports multiple explanation methods, visualization tools, and a unified interface for easy integration.

## Overview

The module includes:
- **SHAP explanations** - Global and local feature importance using Shapley values
- **LIME explanations** - Local interpretable model-agnostic explanations
- **Rule-based explanations** - Domain-specific interpretable rules
- **Attention mechanisms** - For neural network interpretability
- **Visualization tools** - Comprehensive plotting and dashboard creation
- **Caching system** - Performance optimization for repeated explanations

## SHAP Integration

SHAP (SHapley Additive exPlanations) provides model-agnostic explanations based on game theory. It assigns each feature an importance value for a particular prediction.

### Features
- Global feature importance across all predictions
- Local explanations for individual instances
- Support for various model types (sklearn, PyTorch, Stable-Baselines3)
- Efficient kernel-based approximation

### Example Usage
```python
from src.explainability.shap_lime import SHAPExplainer

explainer = SHAPExplainer()
explanation = explainer.explain(
    model=trained_model,
    observation=test_data,
    background_data=training_data,
    feature_names=feature_names,
    nsamples=100
)
```

## LIME Integration

LIME (Local Interpretable Model-agnostic Explanations) explains individual predictions by learning an interpretable model locally around the prediction.

### Features
- Instance-specific explanations
- Model-agnostic approach
- Handles tabular data with feature perturbation
- Configurable number of features to explain

### Example Usage
```python
from src.explainability.shap_lime import LIMEExplainer

explainer = LIMEExplainer()
explanation = explainer.explain(
    model=trained_model,
    observation=single_instance,
    training_data=training_data,
    feature_names=feature_names,
    num_features=10
)
```

## Rule-based Explanations

The rule-based system provides interpretable explanations using domain-specific knowledge about market anomalies.

### Features
- Fast, deterministic explanations
- Domain knowledge integration
- Threshold-based anomaly detection
- Human-readable explanations

### Available Rules
- **High Volatility**: Detects unusual price volatility
- **Price Spike**: Identifies sudden price movements
- **Volume Anomaly**: Flags unusual trading volumes
- **Order Imbalance**: Detects bid-ask imbalances
- **Spread Anomaly**: Identifies unusual bid-ask spreads

### Example Usage
```python
from src.explainability.rule_based import MarketAnomalyRules

rule_system = MarketAnomalyRules()
explanation = rule_system.explain_observation(
    observation=market_data,
    feature_names=feature_names
)

print(f"Anomaly score: {explanation['anomaly_score']}")
print(f"Triggered rules: {explanation['triggered_rules']}")
print(f"Explanation: {explanation['explanation_text']}")
```

## Unified Interface

The unified interface provides a consistent API for all explanation methods.

### Usage
```python
from src.explainability.interface import explain_instance, pretty_print_explanation

# SHAP explanation
shap_result = explain_instance(
    model=model,
    observation=data,
    method='shap',
    feature_names=features,
    nsamples=100
)

# LIME explanation
lime_result = explain_instance(
    model=model,
    observation=data,
    method='lime',
    feature_names=features,
    num_features=10
)

# Rule-based explanation
rule_result = explain_instance(
    model=model,
    observation=data,
    method='rule',
    feature_names=features
)

# Pretty print any explanation
print(pretty_print_explanation(shap_result))
```

## Visualization Tools

Comprehensive visualization tools for explanation results.

### Available Plots
- SHAP summary plots
- LIME feature importance plots
- Attention heatmaps
- Feature importance comparisons
- Interactive dashboards

### Example Usage
```python
from src.explainability.visualization import (
    plot_shap_summary, plot_lime_explanation, 
    create_explanation_dashboard
)

# Plot SHAP results
plot_shap_summary(shap_explanation)

# Plot LIME results
plot_lime_explanation(lime_explanation, instance_idx=0)

# Create comprehensive dashboard
dashboard_data = create_explanation_dashboard(
    [shap_result, lime_result, rule_result],
    save_dir='artifacts/dashboard',
    title='Market Anomaly Explanations'
)
```

## Caching System

The caching system optimizes performance by storing explanation results.

### Features
- Automatic caching of expensive computations
- Hash-based key generation
- JSON serialization with numpy support
- Cache management utilities

### Example Usage
```python
from src.explainability.cache import ExplanationCache

cache = ExplanationCache(cache_dir='artifacts/explanations')

# Store explanation
cache.store('my_explanation', explanation_data)

# Retrieve explanation
cached_result = cache.get('my_explanation')

# Get cache statistics
info = cache.get_cache_info()
print(f"Total entries: {info['total_entries']}")
```

## Configuration

The explainability module can be configured through the main config file:

```yaml
explainability:
  cache_dir: "artifacts/explanations"
  shap:
    nsamples: 100
    background_size: 100
  lime:
    num_features: 10
    num_samples: 1000
  rules:
    volatility_threshold: 2.0
    volume_threshold: 3.0
    spread_threshold: 0.01
```

## Performance Considerations

- **SHAP**: Computationally expensive, use caching for repeated explanations
- **LIME**: Moderate cost, good for individual instance analysis
- **Rule-based**: Very fast, suitable for real-time applications
- **Caching**: Significantly improves performance for repeated queries

## Best Practices

1. **Use appropriate methods**: SHAP for global understanding, LIME for local insights, rules for fast interpretations
2. **Cache results**: Enable caching for expensive computations
3. **Validate explanations**: Cross-check results across different methods
4. **Consider context**: Domain knowledge is crucial for interpretation
5. **Visualize results**: Use provided visualization tools for better understanding

## Troubleshooting

### Common Issues
- **Missing dependencies**: Install SHAP and LIME with `pip install shap lime`
- **Memory issues**: Reduce nsamples for SHAP or background data size
- **Model compatibility**: Ensure model has predict or predict_proba methods
- **Feature name mismatch**: Verify feature names match data dimensions

### Debug Mode
Enable debug logging for detailed information:
```python
import logging
logging.getLogger('src.explainability').setLevel(logging.DEBUG)
```