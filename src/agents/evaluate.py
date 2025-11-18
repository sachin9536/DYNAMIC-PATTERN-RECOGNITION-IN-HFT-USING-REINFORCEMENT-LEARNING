"""Evaluation script for trained RL agents."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# Stable-baselines3 imports
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.envs.market_env import MarketEnv
from src.envs.cvar_wrapper import CVaRWrapper
from src.data.sequence_builder import load_sequences
from src.utils.config_loader import load_config

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def evaluate_policy_detailed(
    model,
    env,
    n_episodes: int = 50,
    deterministic: bool = True
) -> Dict[str, Any]:
    """
    Evaluate policy with detailed metrics collection.
    
    Args:
        model: Trained RL model
        env: Evaluation environment
        n_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic policy
    
    Returns:
        Dictionary with evaluation results
    """
    episode_rewards = []
    episode_lengths = []
    detection_metrics = []
    cvar_estimates = []
    action_overrides = []
    
    logger.info(f"Starting evaluation for {n_episodes} episodes")
    
    for episode in range(n_episodes):
        obs, _ = env.reset()  # Gymnasium returns (obs, info)
        episode_reward = 0.0
        episode_length = 0
        
        # Episode-level metrics
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        overrides_count = 0
        
        done = False
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # Collect detection metrics if available
            if 'raw_reward_components' in info:
                detection_reward = info['raw_reward_components'].get('detection_reward', 0)
                
                # Simple classification based on detection reward
                if detection_reward > 0:
                    true_positives += 1
                elif detection_reward < -0.1:  # Threshold for false positive
                    false_positives += 1
                elif detection_reward < 0:  # Missed detection
                    false_negatives += 1
                else:
                    true_negatives += 1
            
            # Track action overrides
            if info.get('action_overridden', False):
                overrides_count += 1
        
        # Store episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        action_overrides.append(overrides_count)
        
        # Calculate precision and recall for this episode
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        detection_metrics.append({
            'precision': precision,
            'recall': recall,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        })
        
        # Get CVaR estimate if available
        if hasattr(env, 'get_cvar'):
            cvar_estimates.append(env.get_cvar())
        else:
            cvar_estimates.append(0.0)
        
        if (episode + 1) % 10 == 0:
            logger.info(f"Completed {episode + 1}/{n_episodes} episodes")
    
    # Aggregate results
    results = {
        'n_episodes': n_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_precision': np.mean([m['precision'] for m in detection_metrics]),
        'mean_recall': np.mean([m['recall'] for m in detection_metrics]),
        'total_true_positives': sum(m['true_positives'] for m in detection_metrics),
        'total_false_positives': sum(m['false_positives'] for m in detection_metrics),
        'total_true_negatives': sum(m['true_negatives'] for m in detection_metrics),
        'total_false_negatives': sum(m['false_negatives'] for m in detection_metrics),
        'mean_cvar': np.mean(cvar_estimates),
        'total_action_overrides': sum(action_overrides),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'detection_metrics': detection_metrics,
        'cvar_estimates': cvar_estimates,
        'action_overrides': action_overrides
    }
    
    # Calculate overall F1 score
    overall_precision = results['total_true_positives'] / (results['total_true_positives'] + results['total_false_positives']) if (results['total_true_positives'] + results['total_false_positives']) > 0 else 0.0
    overall_recall = results['total_true_positives'] / (results['total_true_positives'] + results['total_false_negatives']) if (results['total_true_positives'] + results['total_false_negatives']) > 0 else 0.0
    
    results['overall_precision'] = overall_precision
    results['overall_recall'] = overall_recall
    results['f1_score'] = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    return results


def save_evaluation_results(results: Dict[str, Any], output_dir: str) -> None:
    """Save evaluation results to CSV and JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed episode results to CSV
    csv_path = output_dir / "eval_results.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'episode', 'reward', 'length', 'precision', 'recall', 
            'cvar', 'action_overrides', 'true_positives', 'false_positives',
            'true_negatives', 'false_negatives'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(results['n_episodes']):
            writer.writerow({
                'episode': i,
                'reward': results['episode_rewards'][i],
                'length': results['episode_lengths'][i],
                'precision': results['detection_metrics'][i]['precision'],
                'recall': results['detection_metrics'][i]['recall'],
                'cvar': results['cvar_estimates'][i],
                'action_overrides': results['action_overrides'][i],
                'true_positives': results['detection_metrics'][i]['true_positives'],
                'false_positives': results['detection_metrics'][i]['false_positives'],
                'true_negatives': results['detection_metrics'][i]['true_negatives'],
                'false_negatives': results['detection_metrics'][i]['false_negatives']
            })
    
    # Save summary results to JSON
    summary_results = {k: v for k, v in results.items() 
                      if k not in ['episode_rewards', 'episode_lengths', 'detection_metrics', 
                                  'cvar_estimates', 'action_overrides']}
    
    json_path = output_dir / "eval_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {csv_path} and {json_path}")


def load_model(model_path: str, algo: str):
    """Load trained model."""
    if algo.lower() == 'ppo':
        model = PPO.load(model_path)
    elif algo.lower() == 'sac':
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    
    logger.info(f"Loaded {algo.upper()} model from {model_path}")
    return model


def create_eval_env(sequences: np.ndarray, targets: np.ndarray, cfg: Dict[str, Any]):
    """Create evaluation environment."""
    # Use test split or full data for evaluation
    env = MarketEnv(sequences, targets, cfg)
    
    # Apply CVaR wrapper if it was used in training
    rl_cfg = cfg.get('rl', {})
    if rl_cfg.get('use_cvar', False):
        cvar_cfg = rl_cfg.get('cvar', {})
        env = CVaRWrapper(
            env,
            alpha=cvar_cfg.get('alpha', 0.95),
            window=cvar_cfg.get('window', 1000),
            penalty_scale=cvar_cfg.get('penalty_scale', 1.0)
        )
    
    return env


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"],
                       help="RL algorithm used")
    parser.add_argument("--sequences_path", type=str,
                       default="data/processed/sequences/sample_sequences.npz",
                       help="Path to sequences file")
    parser.add_argument("--n_episodes", type=int, default=50,
                       help="Number of evaluation episodes")
    parser.add_argument("--out_dir", type=str, default="artifacts/eval/",
                       help="Output directory for evaluation results")
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic policy")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        cfg = load_config(args.config)
        
        # Load sequences and targets
        logger.info(f"Loading sequences from {args.sequences_path}")
        sequences, targets, metadata = load_sequences(args.sequences_path)
        logger.info(f"Loaded {len(sequences)} sequences for evaluation")
        
        # Create evaluation environment
        env = create_eval_env(sequences, targets, cfg)
        
        # Load trained model
        model = load_model(args.model, args.algo)
        
        # Evaluate policy
        results = evaluate_policy_detailed(
            model=model,
            env=env,
            n_episodes=args.n_episodes,
            deterministic=args.deterministic
        )
        
        # Save results
        save_evaluation_results(results, args.out_dir)
        
        # Print summary
        print(f"✅ Evaluation completed successfully!")
        print(f"📊 Episodes: {results['n_episodes']}")
        print(f"🎯 Mean Reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
        print(f"📈 Precision: {results['overall_precision']:.4f}")
        print(f"📉 Recall: {results['overall_recall']:.4f}")
        print(f"🎪 F1 Score: {results['f1_score']:.4f}")
        print(f"⚠️  CVaR: {results['mean_cvar']:.4f}")
        print(f"🔄 Action Overrides: {results['total_action_overrides']}")
        print(f"💾 Results saved to: {args.out_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())