"""CVaR-aware reward wrapper for risk-sensitive RL training."""

import gymnasium as gym
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, Any

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class CVaRWrapper(gym.Wrapper):
    """
    Wrapper that applies CVaR-based risk penalties to rewards.
    
    This is an empirical, online approximation of CVaR that:
    1. Tracks recent episode returns in a sliding window
    2. Computes empirical VaR and CVaR estimates
    3. Applies penalties when step rewards cause drops below VaR threshold
    
    Limitations:
    - Uses empirical quantiles, not true CVaR optimization
    - Online estimation may be noisy with small windows
    - Penalty is applied at step level, not episode level
    - Does not modify advantage estimation in the RL algorithm
    """
    
    def __init__(
        self,
        env: gym.Env,
        alpha: float = 0.95,
        window: int = 1000,
        penalty_scale: float = 1.0
    ):
        """
        Initialize CVaR wrapper.
        
        Args:
            env: Base environment to wrap
            alpha: CVaR confidence level (e.g., 0.95 for 95% CVaR)
            window: Number of recent episode returns to track
            penalty_scale: Scaling factor for CVaR penalty
        """
        super().__init__(env)
        
        self.alpha = alpha
        self.window = window
        self.penalty_scale = penalty_scale
        
        # Track episode returns for CVaR calculation
        self.episode_returns = deque(maxlen=window)
        self.current_episode_return = 0.0
        
        # CVaR estimates
        self.current_var = 0.0
        self.current_cvar = 0.0
        
        # Statistics
        self.total_penalties_applied = 0
        self.total_penalty_amount = 0.0
        
        logger.info(f"CVaRWrapper initialized: alpha={alpha}, window={window}, "
                   f"penalty_scale={penalty_scale}")
    
    def reset(self, **kwargs) -> np.ndarray:
        """Reset environment and episode tracking."""
        # Store previous episode return if we had one
        if hasattr(self, 'current_episode_return'):
            if self.current_episode_return != 0.0:
                self.episode_returns.append(self.current_episode_return)
                self._update_cvar_estimates()
        
        # Reset episode tracking
        self.current_episode_return = 0.0
        
        return self.env.reset(**kwargs)
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute step with CVaR-aware reward modification.
        
        Args:
            action: Action to execute
        
        Returns:
            Tuple of (observation, modified_reward, done, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Track cumulative return
        self.current_episode_return += reward
        
        # Apply CVaR penalty if conditions are met
        cvar_penalty = 0.0
        penalty_applied = False
        
        if len(self.episode_returns) > 10:  # Need some history
            # Check if current return drop exceeds VaR threshold
            if self.current_episode_return < self.current_var:
                # Calculate penalty based on how far below VaR we are
                loss_magnitude = abs(self.current_episode_return - self.current_var)
                cvar_penalty = -self.penalty_scale * loss_magnitude
                penalty_applied = True
                
                self.total_penalties_applied += 1
                self.total_penalty_amount += abs(cvar_penalty)
                
                logger.debug(f"CVaR penalty applied: {cvar_penalty:.4f}, "
                           f"episode_return={self.current_episode_return:.4f}, "
                           f"VaR={self.current_var:.4f}")
        
        # Modify reward
        modified_reward = reward + cvar_penalty
        
        # Update info with CVaR information
        cvar_info = {
            'cvar_penalty': cvar_penalty,
            'penalty_applied': penalty_applied,
            'current_var': self.current_var,
            'current_cvar': self.current_cvar,
            'episode_return': self.current_episode_return,
            'n_episodes_tracked': len(self.episode_returns)
        }
        
        # Merge with existing info
        if isinstance(info, dict):
            info.update(cvar_info)
        else:
            info = cvar_info
        
        # If episode is done, update CVaR estimates
        if done:
            self.episode_returns.append(self.current_episode_return)
            self._update_cvar_estimates()
        
        return obs, modified_reward, terminated, truncated, info
    
    def _update_cvar_estimates(self) -> None:
        """Update VaR and CVaR estimates from recent episode returns."""
        if len(self.episode_returns) < 5:
            return
        
        returns_array = np.array(self.episode_returns)
        
        # Calculate VaR (Value at Risk) - the alpha quantile
        self.current_var = np.quantile(returns_array, 1 - self.alpha)
        
        # Calculate CVaR (Conditional Value at Risk) - mean of returns below VaR
        tail_returns = returns_array[returns_array <= self.current_var]
        if len(tail_returns) > 0:
            self.current_cvar = np.mean(tail_returns)
        else:
            self.current_cvar = self.current_var
        
        logger.debug(f"Updated CVaR estimates: VaR={self.current_var:.4f}, "
                    f"CVaR={self.current_cvar:.4f} (from {len(self.episode_returns)} episodes)")
    
    def get_cvar(self) -> float:
        """
        Get current CVaR estimate.
        
        Returns:
            Current CVaR estimate (mean of tail losses)
        """
        return self.current_cvar
    
    def get_var(self) -> float:
        """
        Get current VaR estimate.
        
        Returns:
            Current VaR estimate (alpha quantile)
        """
        return self.current_var
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get CVaR wrapper statistics.
        
        Returns:
            Dictionary with wrapper statistics
        """
        return {
            'total_penalties_applied': self.total_penalties_applied,
            'total_penalty_amount': self.total_penalty_amount,
            'current_var': self.current_var,
            'current_cvar': self.current_cvar,
            'episodes_tracked': len(self.episode_returns),
            'alpha': self.alpha,
            'window': self.window,
            'penalty_scale': self.penalty_scale
        }
    
    def reset_statistics(self) -> None:
        """Reset penalty statistics."""
        self.total_penalties_applied = 0
        self.total_penalty_amount = 0.0


if __name__ == "__main__":
    # Test CVaR wrapper
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    try:
        from src.envs.market_env import MarketEnv
        from src.data.sequence_builder import load_sequences
        
        # Load sample sequences
        seq_path = "data/processed/sequences/sample_sequences.npz"
        if Path(seq_path).exists():
            sequences, targets, metadata = load_sequences(seq_path)
            
            # Create base environment
            cfg = {'episode_length': 20, 'flatten_obs': False}
            base_env = MarketEnv(sequences, targets, cfg)
            
            # Wrap with CVaR
            env = CVaRWrapper(base_env, alpha=0.9, window=50, penalty_scale=0.5)
            
            print("Testing CVaR wrapper...")
            print(f"Initial CVaR: {env.get_cvar():.4f}")
            
            # Run multiple episodes to build history
            for episode in range(10):
                obs = env.reset()
                episode_reward = 0
                
                for step in range(15):
                    action = env.action_space.sample()
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    
                    if done:
                        break
                
                print(f"Episode {episode}: reward={episode_reward:.4f}, "
                      f"CVaR={env.get_cvar():.4f}, VaR={env.get_var():.4f}")
                
                if 'cvar_penalty' in info:
                    print(f"  CVaR penalty: {info['cvar_penalty']:.4f}")
            
            # Print final statistics
            stats = env.get_statistics()
            print(f"\nFinal statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
        else:
            print(f"Sample sequences not found at {seq_path}")
            
    except Exception as e:
        print(f"Error testing CVaR wrapper: {e}")
        import traceback
        traceback.print_exc()