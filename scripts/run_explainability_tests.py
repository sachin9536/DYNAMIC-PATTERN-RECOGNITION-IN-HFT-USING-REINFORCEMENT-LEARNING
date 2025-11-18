#!/usr/bin/env python3
"""
Comprehensive test script for explainability functionality.
Tests all explanation methods and visualization tools.
"""

import sys
import os
from pathlib import Path
import numpy as np
import json
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.explainability.interface import explain_instance, pretty_print_explanation
    from src.explainability.visualization import (
        plot_shap_summary, plot_lime_explanation, plot_attention_heatmap,
        create_explanation_dashboard, plot_feature_importance_comparison
    )
    from src.explainability.rule_based import MarketAnomalyRules
    from src.explainability.shap_lime import explain_with_shap, explain_with_lime
    from src.explainability.attention_utils import register_attention_hooks, aggregate_attention
    from src.data.synthetic_market_data import SyntheticMarketDataGenerator
    from src.data.preprocessing import MarketDataPreprocessor
    from src.utils.logger import get_logger
    
    logger = get_logger(__name__)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def test_data_generation():
    """Test synthetic data generation."""
    print("🧪 Testing data generation...")
    
    try:
        # Generate test data
        generator = SyntheticMarketDataGenerator()
        data = generator.generate_market_session(
            n_samples=100,
            anomaly_probability=0.2,
            random_state=42
        )
        
        # Preprocess data
        preprocessor = MarketDataPreprocessor()
        processed_data = preprocessor.preprocess_data(data)
        
        print(f"✅ Generated {len(data)} samples with {processed_data.shape[1]} features")
        print(f"✅ Anomaly rate: {data['is_anomaly'].mean():.2%}")
        
        return processed_data, data['is_anomaly'].values, preprocessor.feature_names
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        traceback.print_exc()
        return None, None, None


def test_simple_model():
    """Test with a simple sklearn model."""
    print("🧪 Testing simple model training...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        # Get test data
        X, y, feature_names = test_data_generation()
        if X is None:
            return None, None, None
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        print(f"✅ Model trained with accuracy: {accuracy:.3f}")
        
        return model, X_test, feature_names
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        traceback.print_exc()
        return None, None, None


def test_rule_based_explanations():
    """Test rule-based explanation system."""
    print("🧪 Testing rule-based explanations...")
    
    try:
        # Initialize rule system
        rule_system = MarketAnomalyRules()
        
        # Test with normal observation
        normal_obs = {
            'volume_change': 0.5,
            'price_volatility': 1.0,
            'order_imbalance': 0.3,
            'spread': 0.8,
            'momentum': 0.2
        }
        
        result = rule_system.explain_observation(normal_obs)
        print(f"✅ Normal observation - Anomaly score: {result['anomaly_score']:.3f}")
        print(f"✅ Triggered rules: {len(result['triggered_rules'])}")
        
        # Test with anomalous observation
        anomalous_obs = {
            'volume_change': 4.0,
            'price_volatility': 3.5,
            'order_imbalance': 0.8,
            'spread': 2.5,
            'momentum': 3.0
        }
        
        result = rule_system.explain_observation(anomalous_obs)
        print(f"✅ Anomalous observation - Anomaly score: {result['anomaly_score']:.3f}")
        print(f"✅ Triggered rules: {len(result['triggered_rules'])}")
        
        # Test rule summary
        summary = rule_system.get_rule_summary()
        print(f"✅ Rule system has {summary['total_rules']} rules")
        
        return True
        
    except Exception as e:
        print(f"❌ Rule-based explanation failed: {e}")
        traceback.print_exc()
        return False


def test_shap_explanations():
    """Test SHAP explanations."""
    print("🧪 Testing SHAP explanations...")
    
    try:
        # Get model and test data
        model, X_test, feature_names = test_simple_model()
        if model is None:
            print("⚠️  Skipping SHAP test - no model available")
            return False
        
        # Test SHAP explanation
        test_sample = X_test[:3]  # First 3 samples
        
        explanation = explain_instance(
            model=model,
            observation=test_sample,
            method='shap',
            feature_names=feature_names,
            nsamples=20,  # Small for testing
            cache=False
        )
        
        if 'error' in explanation:
            print(f"⚠️  SHAP explanation had error: {explanation['error']}")
            return False
        
        if 'shap_values' in explanation:
            shap_values = np.array(explanation['shap_values'])
            print(f"✅ SHAP values shape: {shap_values.shape}")
            print(f"✅ Base value: {explanation.get('base_values', 'N/A')}")
            
            # Test visualization
            try:
                plot_meta = plot_shap_summary(explanation)
                print(f"✅ SHAP visualization created: {plot_meta['plot_type']}")
            except Exception as viz_e:
                print(f"⚠️  SHAP visualization failed: {viz_e}")
            
            return True
        else:
            print("⚠️  No SHAP values in explanation")
            return False
        
    except Exception as e:
        print(f"❌ SHAP explanation failed: {e}")
        traceback.print_exc()
        return False


def test_lime_explanations():
    """Test LIME explanations."""
    print("🧪 Testing LIME explanations...")
    
    try:
        # Get model and test data
        model, X_test, feature_names = test_simple_model()
        if model is None:
            print("⚠️  Skipping LIME test - no model available")
            return False
        
        # Test LIME explanation
        test_sample = X_test[:2]  # First 2 samples
        
        explanation = explain_instance(
            model=model,
            observation=test_sample,
            method='lime',
            feature_names=feature_names,
            num_features=5,
            cache=False
        )
        
        if 'error' in explanation:
            print(f"⚠️  LIME explanation had error: {explanation['error']}")
            return False
        
        if 'explanations' in explanation:
            explanations = explanation['explanations']
            print(f"✅ LIME explanations for {len(explanations)} instances")
            
            # Check first explanation
            if explanations and 'feature_importance' in explanations[0]:
                importance = explanations[0]['feature_importance']
                print(f"✅ First instance has {len(importance)} feature importances")
                
                # Test visualization
                try:
                    plot_meta = plot_lime_explanation(explanation, instance_idx=0)
                    print(f"✅ LIME visualization created: {plot_meta['plot_type']}")
                except Exception as viz_e:
                    print(f"⚠️  LIME visualization failed: {viz_e}")
                
                return True
            else:
                print("⚠️  No feature importance in LIME explanation")
                return False
        else:
            print("⚠️  No explanations in LIME result")
            return False
        
    except Exception as e:
        print(f"❌ LIME explanation failed: {e}")
        traceback.print_exc()
        return False


def test_attention_explanations():
    """Test attention-based explanations."""
    print("🧪 Testing attention explanations...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Create a simple model with attention
        class SimpleAttentionModel(nn.Module):
            def __init__(self, input_size=10, hidden_size=32):
                super().__init__()
                self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
                self.linear1 = nn.Linear(input_size, hidden_size)
                self.linear2 = nn.Linear(hidden_size, 2)
                
            def forward(self, x):
                # Reshape input for attention
                if x.ndim == 2:
                    x = x.unsqueeze(1)  # Add sequence dimension
                
                x = self.linear1(x)
                attn_out, attn_weights = self.attention(x, x, x)
                x = self.linear2(attn_out.mean(dim=1))  # Pool sequence dimension
                return x
        
        model = SimpleAttentionModel()
        model.eval()
        
        # Test attention explanation
        test_input = torch.randn(1, 10)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        explanation = explain_instance(
            model=model,
            observation=test_input.numpy(),
            method='attention',
            feature_names=feature_names,
            cache=False
        )
        
        if 'error' in explanation:
            print(f"⚠️  Attention explanation had error: {explanation['error']}")
            return False
        
        if 'attention_weights' in explanation:
            attention_shape = explanation.get('attention_shape', 'unknown')
            print(f"✅ Attention weights shape: {attention_shape}")
            
            # Test visualization
            try:
                plot_meta = plot_attention_heatmap(explanation)
                print(f"✅ Attention visualization created: {plot_meta['plot_type']}")
            except Exception as viz_e:
                print(f"⚠️  Attention visualization failed: {viz_e}")
            
            return True
        else:
            print("⚠️  No attention weights in explanation")
            return False
        
    except ImportError:
        print("⚠️  PyTorch not available - skipping attention test")
        return False
    except Exception as e:
        print(f"❌ Attention explanation failed: {e}")
        traceback.print_exc()
        return False


def test_unified_interface():
    """Test the unified explanation interface."""
    print("🧪 Testing unified interface...")
    
    try:
        # Get model and test data
        model, X_test, feature_names = test_simple_model()
        if model is None:
            print("⚠️  Skipping unified interface test - no model available")
            return False
        
        test_sample = X_test[0]
        methods_to_test = ['rule']  # Start with rule-based as it's most reliable
        
        # Add other methods if their dependencies are available
        try:
            import shap
            methods_to_test.append('shap')
        except ImportError:
            pass
        
        try:
            import lime
            methods_to_test.append('lime')
        except ImportError:
            pass
        
        explanations = []
        for method in methods_to_test:
            try:
                explanation = explain_instance(
                    model=model,
                    observation=test_sample,
                    method=method,
                    feature_names=feature_names,
                    cache=False,
                    nsamples=10 if method == 'shap' else None,
                    num_features=5 if method == 'lime' else None
                )
                
                explanations.append(explanation)
                print(f"✅ {method.upper()} explanation successful")
                
                # Test pretty printing
                pretty_output = pretty_print_explanation(explanation)
                print(f"✅ Pretty print for {method} successful")
                
            except Exception as e:
                print(f"⚠️  {method.upper()} explanation failed: {e}")
        
        if explanations:
            print(f"✅ Unified interface tested with {len(explanations)} methods")
            return True
        else:
            print("❌ No explanations succeeded")
            return False
        
    except Exception as e:
        print(f"❌ Unified interface test failed: {e}")
        traceback.print_exc()
        return False


def test_dashboard_creation():
    """Test dashboard creation."""
    print("🧪 Testing dashboard creation...")
    
    try:
        # Create dummy explanations for testing
        dummy_explanations = [
            {
                'method': 'rule',
                'triggered_rules': ['volume_spike', 'high_volatility'],
                'anomaly_score': 0.8,
                'explanation_text': 'Multiple anomalies detected',
                'feature_names': [f'feature_{i}' for i in range(5)]
            },
            {
                'method': 'shap',
                'shap_values': np.random.randn(10, 5).tolist(),
                'base_values': 0.1,
                'feature_names': [f'feature_{i}' for i in range(5)]
            }
        ]
        
        # Test dashboard creation
        dashboard_dir = project_root / "artifacts" / "explanations" / "test_dashboard"
        dashboard_data = create_explanation_dashboard(
            dummy_explanations,
            save_dir=str(dashboard_dir),
            title="Test Dashboard"
        )
        
        print(f"✅ Dashboard created in {dashboard_dir}")
        print(f"✅ Dashboard summary: {dashboard_data['summary']}")
        
        # Test comparison plot
        try:
            comparison_meta = plot_feature_importance_comparison(dummy_explanations)
            print(f"✅ Feature comparison plot created: {comparison_meta['plot_type']}")
        except Exception as comp_e:
            print(f"⚠️  Comparison plot failed: {comp_e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard creation failed: {e}")
        traceback.print_exc()
        return False


def test_cli_functionality():
    """Test CLI functionality."""
    print("🧪 Testing CLI functionality...")
    
    try:
        from src.explainability.cli import load_model, load_data
        
        # Create test data file
        test_data_dir = project_root / "artifacts" / "test_data"
        test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save test data
        test_data = np.random.randn(10, 5)
        test_data_path = test_data_dir / "test_data.npy"
        np.save(test_data_path, test_data)
        
        # Test data loading
        loaded_data = load_data(str(test_data_path))
        print(f"✅ CLI data loading successful: {loaded_data.shape}")
        
        # Test with CSV
        import pandas as pd
        df = pd.DataFrame(test_data, columns=[f'feature_{i}' for i in range(5)])
        csv_path = test_data_dir / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        loaded_csv = load_data(str(csv_path))
        print(f"✅ CLI CSV loading successful: {loaded_csv.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI functionality test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all explainability tests."""
    print("🚀 Starting comprehensive explainability tests...\n")
    
    test_results = {}
    
    # Run individual tests
    tests = [
        ("Data Generation", test_data_generation),
        ("Rule-based Explanations", test_rule_based_explanations),
        ("SHAP Explanations", test_shap_explanations),
        ("LIME Explanations", test_lime_explanations),
        ("Attention Explanations", test_attention_explanations),
        ("Unified Interface", test_unified_interface),
        ("Dashboard Creation", test_dashboard_creation),
        ("CLI Functionality", test_cli_functionality),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            test_results[test_name] = result
            
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"⚠️  {test_name} FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            test_results[test_name] = False
            traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All explainability tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)