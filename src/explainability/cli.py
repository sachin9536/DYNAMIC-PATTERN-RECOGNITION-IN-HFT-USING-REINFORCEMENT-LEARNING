"""Command-line interface for explainability tools."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

try:
    from src.explainability.interface import explain_instance, pretty_print_explanation
    from src.explainability.visualization import (
        create_explanation_dashboard,
        plot_feature_importance_comparison
    )
    from src.explainability.rule_based import MarketAnomalyRules
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def load_model(model_path: str):
    """Load a trained model from file."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Try different model loading methods
    if model_path.suffix == '.pkl':
        import pickle
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    elif model_path.suffix in ['.pt', '.pth']:
        import torch
        return torch.load(model_path, map_location='cpu')
    
    elif model_path.suffix == '.zip':
        # Stable-baselines3 model
        try:
            from stable_baselines3 import PPO, SAC
            # Try PPO first
            try:
                return PPO.load(str(model_path))
            except:
                return SAC.load(str(model_path))
        except ImportError:
            raise ImportError("stable-baselines3 not available for loading .zip models")
    
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")


def load_data(data_path: str) -> np.ndarray:
    """Load observation data from file."""
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if data_path.suffix == '.npy':
        return np.load(data_path)
    
    elif data_path.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(data_path)
        return df.values
    
    elif data_path.suffix == '.json':
        with open(data_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            return np.array(data)
        elif isinstance(data, dict) and 'observations' in data:
            return np.array(data['observations'])
        else:
            raise ValueError("JSON data format not recognized")
    
    else:
        raise ValueError(f"Unsupported data format: {data_path.suffix}")


def explain_command(args):
    """Handle the explain command."""
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    
    print(f"Loading data from {args.data}...")
    data = load_data(args.data)
    
    # Handle single observation vs multiple
    if data.ndim == 1:
        observations = data.reshape(1, -1)
    else:
        observations = data
    
    # Limit number of observations if specified
    if args.max_samples and observations.shape[0] > args.max_samples:
        observations = observations[:args.max_samples]
        print(f"Limited to {args.max_samples} observations")
    
    # Load feature names if provided
    feature_names = None
    if args.feature_names:
        feature_path = Path(args.feature_names)
        if feature_path.exists():
            if feature_path.suffix == '.json':
                with open(feature_path, 'r') as f:
                    feature_names = json.load(f)
            elif feature_path.suffix == '.txt':
                with open(feature_path, 'r') as f:
                    feature_names = [line.strip() for line in f]
        else:
            # Treat as comma-separated string
            feature_names = args.feature_names.split(',')
    
    # Run explanations
    explanations = []
    methods = args.methods.split(',')
    
    print(f"Running explanations using methods: {methods}")
    
    for method in methods:
        method = method.strip()
        print(f"\n--- Running {method.upper()} explanation ---")
        
        try:
            # For efficiency, explain only first observation for SHAP/LIME
            if method in ['shap', 'lime'] and observations.shape[0] > 1:
                obs_to_explain = observations[:1]
            else:
                obs_to_explain = observations[0] if observations.shape[0] == 1 else observations[0]
            
            explanation = explain_instance(
                model=model,
                observation=obs_to_explain,
                method=method,
                feature_names=feature_names,
                cache=not args.no_cache,
                cache_dir=args.cache_dir,
                nsamples=args.shap_samples if method == 'shap' else None,
                num_features=args.lime_features if method == 'lime' else None
            )
            
            explanations.append(explanation)
            
            # Print results
            print(pretty_print_explanation(explanation))
            
        except Exception as e:
            print(f"❌ {method.upper()} explanation failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(explanations, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {output_path}")
    
    # Create visualizations if requested
    if args.visualize and explanations:
        viz_dir = Path(args.viz_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Creating visualizations in {viz_dir}...")
        
        try:
            dashboard_data = create_explanation_dashboard(
                explanations,
                save_dir=str(viz_dir),
                title=f"Explanations for {args.model}"
            )
            print(f"Created dashboard with {dashboard_data['summary']['successful_plots']} plots")
            
            # Create comparison plot if multiple methods
            if len(explanations) > 1:
                comparison_path = viz_dir / "feature_importance_comparison.png"
                comparison_meta = plot_feature_importance_comparison(
                    explanations,
                    save_path=str(comparison_path)
                )
                print(f"Created feature importance comparison plot")
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()


def rules_command(args):
    """Handle the rules command."""
    rule_system = MarketAnomalyRules()
    
    if args.update_thresholds:
        # Parse threshold updates
        threshold_updates = {}
        for update in args.update_thresholds.split(','):
            key, value = update.split('=')
            threshold_updates[key.strip()] = float(value.strip())
        
        rule_system.update_thresholds(threshold_updates)
        print(f"Updated thresholds: {threshold_updates}")
    
    if args.data:
        print(f"Loading data from {args.data}...")
        data = load_data(args.data)
        
        # Handle single observation vs multiple
        if data.ndim == 1:
            observations = [data]
        else:
            observations = data[:args.max_samples] if args.max_samples else data
        
        # Load feature names if provided
        feature_names = None
        if args.feature_names:
            feature_path = Path(args.feature_names)
            if feature_path.exists():
                if feature_path.suffix == '.json':
                    with open(feature_path, 'r') as f:
                        feature_names = json.load(f)
                elif feature_path.suffix == '.txt':
                    with open(feature_path, 'r') as f:
                        feature_names = [line.strip() for line in f]
            else:
                feature_names = args.feature_names.split(',')
        
        # Analyze observations
        results = []
        for i, obs in enumerate(observations):
            print(f"\n--- Analyzing observation {i+1} ---")
            
            result = rule_system.explain_observation(obs, feature_names)
            results.append(result)
            
            print(f"Anomaly score: {result['anomaly_score']:.3f}")
            print(f"Triggered rules: {len(result['triggered_rules'])}")
            print(f"Explanation: {result['explanation_text']}")
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Results saved to {output_path}")
    
    # Show rule summary
    if args.summary:
        print("\n--- Rule System Summary ---")
        summary = rule_system.get_rule_summary()
        print(f"Total rules: {summary['total_rules']}")
        print(f"Rule types: {summary['rule_types']}")
        print(f"Thresholds: {summary['thresholds']}")


def dashboard_command(args):
    """Handle the dashboard command."""
    # Load explanation results
    explanations = []
    
    for result_file in args.results:
        result_path = Path(result_file)
        if not result_path.exists():
            print(f"⚠️  Result file not found: {result_file}")
            continue
        
        with open(result_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            explanations.extend(data)
        else:
            explanations.append(data)
    
    if not explanations:
        print("❌ No valid explanation results found")
        return
    
    print(f"Creating dashboard for {len(explanations)} explanations...")
    
    # Create dashboard
    dashboard_data = create_explanation_dashboard(
        explanations,
        save_dir=args.output_dir,
        title=args.title
    )
    
    print(f"✅ Dashboard created in {args.output_dir}")
    print(f"Generated {dashboard_data['summary']['successful_plots']} plots")
    
    # Create comparison plot if multiple methods
    methods_used = dashboard_data['summary']['methods_used']
    if len(methods_used) > 1:
        comparison_path = Path(args.output_dir) / "feature_importance_comparison.png"
        try:
            comparison_meta = plot_feature_importance_comparison(
                explanations,
                save_path=str(comparison_path)
            )
            print(f"Created feature importance comparison plot")
        except Exception as e:
            print(f"⚠️  Failed to create comparison plot: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Explainability tools for market anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explain model predictions using SHAP
  python -m src.explainability.cli explain --model model.pkl --data data.npy --methods shap
  
  # Use multiple explanation methods with visualization
  python -m src.explainability.cli explain --model model.zip --data data.csv --methods shap,lime,rule --visualize
  
  # Analyze data using rule-based system only
  python -m src.explainability.cli rules --data data.npy --summary
  
  # Create dashboard from saved results
  python -m src.explainability.cli dashboard --results result1.json result2.json --output-dir dashboard/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Explain command
    explain_parser = subparsers.add_parser('explain', help='Explain model predictions')
    explain_parser.add_argument('--model', required=True, help='Path to trained model file')
    explain_parser.add_argument('--data', required=True, help='Path to observation data')
    explain_parser.add_argument('--methods', default='shap,lime,rule', 
                               help='Comma-separated explanation methods (shap,lime,attention,rule)')
    explain_parser.add_argument('--feature-names', help='Path to feature names file or comma-separated names')
    explain_parser.add_argument('--max-samples', type=int, default=10, 
                               help='Maximum number of samples to explain')
    explain_parser.add_argument('--output', help='Path to save explanation results (JSON)')
    explain_parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    explain_parser.add_argument('--viz-dir', default='artifacts/explanations/viz', 
                               help='Directory for visualizations')
    explain_parser.add_argument('--cache-dir', default='artifacts/explanations/cache', 
                               help='Directory for caching')
    explain_parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    explain_parser.add_argument('--shap-samples', type=int, default=100, 
                               help='Number of samples for SHAP estimation')
    explain_parser.add_argument('--lime-features', type=int, default=10, 
                               help='Number of features for LIME explanation')
    explain_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Rules command
    rules_parser = subparsers.add_parser('rules', help='Rule-based analysis')
    rules_parser.add_argument('--data', help='Path to observation data')
    rules_parser.add_argument('--feature-names', help='Path to feature names file or comma-separated names')
    rules_parser.add_argument('--max-samples', type=int, default=100, 
                             help='Maximum number of samples to analyze')
    rules_parser.add_argument('--update-thresholds', 
                             help='Update thresholds (format: key1=value1,key2=value2)')
    rules_parser.add_argument('--summary', action='store_true', help='Show rule system summary')
    rules_parser.add_argument('--output', help='Path to save analysis results (JSON)')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Create explanation dashboard')
    dashboard_parser.add_argument('--results', nargs='+', required=True, 
                                 help='Paths to explanation result files')
    dashboard_parser.add_argument('--output-dir', default='artifacts/explanations/dashboard', 
                                 help='Output directory for dashboard')
    dashboard_parser.add_argument('--title', default='Model Explanations Dashboard', 
                                 help='Dashboard title')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'explain':
            explain_command(args)
        elif args.command == 'rules':
            rules_command(args)
        elif args.command == 'dashboard':
            dashboard_command(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()