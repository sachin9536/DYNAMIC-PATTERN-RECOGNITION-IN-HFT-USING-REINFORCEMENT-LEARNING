"""Training harness for PPO and SAC agents with CVaR risk management."""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch

# Stable-baselines3 imports
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

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


def seed_everything(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def make_model(algo: str, env, cfg: Dict[str, Any]):
    """
    Create RL model based on algorithm type.
    
    Args:
        algo: Algorithm name ('ppo' or 'sac')
        env: Training environment
        cfg: Configuration dictionary
    
    Returns:
        Initialized RL model
    """
    rl_cfg = cfg.get('rl', {})
    seed = rl_cfg.get('seed', 42)
    
    # Common parameters
    common_params = {
        'env': env,
        'verbose': 1,
        'seed': seed,
        'tensorboard_log': rl_cfg.get('tensorboard_log', 'artifacts/tb_logs')
    }
    
    if algo.lower() == 'ppo':
        model = PPO(
            policy='MlpPolicy',
            learning_rate=rl_cfg.get('learning_rate', 3e-4),
            n_steps=rl_cfg.get('n_steps', 2048),
            batch_size=rl_cfg.get('batch_size', 64),
            n_epochs=rl_cfg.get('n_epochs', 10),
            gamma=rl_cfg.get('gamma', 0.99),
            gae_lambda=rl_cfg.get('gae_lambda', 0.95),
            clip_range=rl_cfg.get('clip_range', 0.2),
            ent_coef=rl_cfg.get('ent_coef', 0.0),
            **common_params
        )
        
    elif algo.lower() == 'sac':
        model = SAC(
            policy='MlpPolicy',
            learning_rate=rl_cfg.get('learning_rate', 3e-4),
            buffer_size=rl_cfg.get('buffer_size', 1000000),
            learning_starts=rl_cfg.get('learning_starts', 100),
            batch_size=rl_cfg.get('batch_size', 256),
            tau=rl_cfg.get('tau', 0.005),
            gamma=rl_cfg.get('gamma', 0.99),
            train_freq=rl_cfg.get('train_freq', 1),
            gradient_steps=rl_cfg.get('gradient_steps', 1),
            **common_params
        )
        
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    
    logger.info(f"Created {algo.upper()} model with policy: MlpPolicy")
    return model


def create_env(sequences: np.ndarray, targets: np.ndarray, cfg: Dict[str, Any], rule_module=None):
    """Create training environment with optional CVaR wrapper."""
    # Create base environment
    env = MarketEnv(sequences, targets, cfg, rule_module)
    
    # Wrap with Monitor for logging
    env = Monitor(env)
    
    # Apply CVaR wrapper if enabled
    rl_cfg = cfg.get('rl', {})
    if rl_cfg.get('use_cvar', False):
        cvar_cfg = rl_cfg.get('cvar', {})
        env = CVaRWrapper(
            env,
            alpha=cvar_cfg.get('alpha', 0.95),
            window=cvar_cfg.get('window', 1000),
            penalty_scale=cvar_cfg.get('penalty_scale', 1.0)
        )
        logger.info("Applied CVaR wrapper to environment")
    
    return env


def train(
    model,
    env,
    total_timesteps: int,
    output_dir: str,
    checkpoint_freq: int = 10000,
    eval_freq: int = 5000
) -> str:
    """
    Train the RL model.
    
    Args:
        model: RL model to train
        env: Training environment
        total_timesteps: Total training timesteps
        output_dir: Directory to save outputs
        checkpoint_freq: Frequency of model checkpoints
        eval_freq: Frequency of evaluation
    
    Returns:
        Path to final saved model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="model"
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback (using same env for simplicity)
    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    logger.info(f"Starting training for {total_timesteps} timesteps")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = output_dir / "final_model"
    model.save(str(final_model_path))
    
    logger.info(f"Training completed. Final model saved to: {final_model_path}")
    
    return str(final_model_path)


def save_training_metadata(cfg: Dict[str, Any], output_dir: str, model_path: str) -> None:
    """Save training metadata and configuration."""
    metadata = {
        'config': cfg,
        'model_path': model_path,
        'seed': cfg.get('rl', {}).get('seed', 42)
    }
    
    metadata_path = Path(output_dir) / "training_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Training metadata saved to: {metadata_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL agent on market data")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"],
                       help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=10000,
                       help="Total training timesteps")
    parser.add_argument("--out_dir", type=str, default="artifacts/",
                       help="Output directory for models and logs")
    parser.add_argument("--sequences_path", type=str, 
                       default="data/processed/sequences/sample_sequences.npz",
                       help="Path to sequences file")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        cfg = load_config(args.config)
        
        # Set seed for reproducibility
        seed = cfg.get('rl', {}).get('seed', 42)
        seed_everything(seed)
        logger.info(f"Set random seed to {seed}")
        
        # Load sequences and targets
        logger.info(f"Loading sequences from {args.sequences_path}")
        sequences, targets, metadata = load_sequences(args.sequences_path)
        logger.info(f"Loaded {len(sequences)} sequences with shape {sequences.shape}")
        
        # Create environment
        env = create_env(sequences, targets, cfg)
        logger.info(f"Created environment: {env}")
        
        # Create model
        model = make_model(args.algo, env, cfg)
        
        # Train model
        model_path = train(
            model=model,
            env=env,
            total_timesteps=args.timesteps,
            output_dir=args.out_dir,
            checkpoint_freq=cfg.get('rl', {}).get('checkpoint_freq', 5000)
        )
        
        # Save metadata
        save_training_metadata(cfg, args.out_dir, model_path)
        
        print(f"✅ Training completed successfully!")
        print(f"📊 Trained for {args.timesteps} timesteps")
        print(f"🤖 Algorithm: {args.algo.upper()}")
        print(f"💾 Model saved to: {model_path}")
        print(f"📁 Output directory: {args.out_dir}")
        
        # Print CVaR statistics if wrapper was used
        if hasattr(env, 'get_statistics'):
            stats = env.get_statistics()
            print(f"📈 CVaR Statistics:")
            print(f"   Penalties applied: {stats['total_penalties_applied']}")
            print(f"   Total penalty amount: {stats['total_penalty_amount']:.4f}")
            print(f"   Current CVaR: {stats['current_cvar']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())