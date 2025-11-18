"""Gym-compatible market simulation environment for RL training."""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from collections import deque
import logging

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class MarketEnv(gym.Env):
    """
    Gym-compatible environment for market anomaly detection and trading.
    
    Action Space: Discrete(3)
        0: Hold (no action)
        1: Signal anomaly
        2: Signal trade
    
    Observation Space: Box representing market sequence features
    """
    
    def __init__(
        self,
        sequences: np.ndarray,
        targets: Optional[np.ndarray] = None,
        cfg: Optional[Dict[str, Any]] = None,
        rule_module: Optional[Any] = None
    ):
        """
        Initialize market environment.
        
        Args:
            sequences: Array of shape (N, seq_len, n_features)
            targets: Optional anomaly/trading targets aligned per sequence
            cfg: Configuration dictionary
            rule_module: Optional module with rule checking functions
        """
        super().__init__()
        
        self.sequences = sequences.astype(np.float32)
        self.targets = targets.astype(np.float32) if targets is not None else None
        self.cfg = cfg or {}
        self.rule_module = rule_module
        
        # Environment parameters
        self.n_sequences, self.seq_len, self.n_features = sequences.shape
        self.flatten_obs = self.cfg.get('flatten_obs', False)
        self.episode_length = self.cfg.get('episode_length', 100)
        self.step_size = self.cfg.get('step_size', 1)
        
        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(3)  # [hold, signal_anomaly, signal_trade]
        
        if self.flatten_obs:
            obs_shape = (self.seq_len * self.n_features,)
        else:
            obs_shape = (self.seq_len, self.n_features)
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.current_sequence_idx = 0
        self.episode_reward = 0.0
        self.episode_returns = deque(maxlen=1000)  # For CVaR calculation
        self.position = 0  # Trading position: -1, 0, 1
        
        # Reward components configuration
        self.reward_cfg = self.cfg.get('rewards', {})
        self.detection_reward_scale = self.reward_cfg.get('detection_scale', 1.0)
        self.trading_reward_scale = self.reward_cfg.get('trading_scale', 0.1)
        self.risk_penalty_scale = self.reward_cfg.get('risk_penalty_scale', 0.5)
        
        # Random state
        self.np_random = None
        self.seed()
        
        logger.info(f"MarketEnv initialized: {self.n_sequences} sequences, "
                   f"obs_shape={obs_shape}, episode_length={self.episode_length}")
    
    def seed(self, seed: Optional[int] = None) -> list:
        """Set random seed for reproducibility."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None, options=None) -> np.ndarray:
        """Reset environment and return initial observation."""
        if seed is not None:
            self.seed(seed)
        
        # Randomly select starting sequence
        self.current_sequence_idx = self.np_random.integers(0, self.n_sequences)
        self.current_step = 0
        self.episode_reward = 0.0
        self.position = 0
        
        # Get initial observation
        obs = self._get_observation()
        
        logger.debug(f"Reset: sequence_idx={self.current_sequence_idx}, obs_shape={obs.shape}")
        
        # Return observation and info dict (required by gymnasium)
        info = {
            'sequence_idx': self.current_sequence_idx,
            'episode_step': 0
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Action to take (0=hold, 1=signal_anomaly, 2=signal_trade)
        
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        
        # Store original action for fusion
        original_action = action
        
        # Apply rule-based fusion if available
        rule_flags = {}
        overridden = False
        triggered_rule = None
        
        if self.rule_module is not None:
            obs = self._get_observation()
            action, overridden, triggered_rule = self._apply_rule_fusion(
                action, obs, rule_flags
            )
        
        # Calculate reward components
        reward_components = self._calculate_reward(action)
        total_reward = sum(reward_components.values())
        
        # Update episode state
        self.current_step += self.step_size
        self.episode_reward += total_reward
        
        # Check if episode is done
        done = (
            self.current_step >= self.episode_length or
            self.current_sequence_idx >= self.n_sequences - 1
        )
        
        if done:
            self.episode_returns.append(self.episode_reward)
        
        # Move to next sequence if needed
        if self.current_step < self.episode_length and not done:
            self.current_sequence_idx = min(
                self.current_sequence_idx + 1,
                self.n_sequences - 1
            )
        
        # Get next observation
        next_obs = self._get_observation()
        
        # Prepare info dictionary
        info = {
            'rule_flags': rule_flags,
            'raw_reward_components': reward_components,
            'original_action': original_action,
            'final_action': action,
            'action_overridden': overridden,
            'triggered_rule': triggered_rule,
            'current_step': self.current_step,
            'sequence_idx': self.current_sequence_idx,
            'position': self.position,
            'episode_reward': self.episode_reward
        }
        
        # For gymnasium compatibility, return terminated and truncated separately
        terminated = done
        truncated = False  # We don't use truncation in this environment
        
        return next_obs, total_reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_sequence_idx >= self.n_sequences:
            # Return zeros if we've exceeded available sequences
            obs = np.zeros((self.seq_len, self.n_features), dtype=np.float32)
        else:
            obs = self.sequences[self.current_sequence_idx].copy()
        
        if self.flatten_obs:
            obs = obs.flatten()
        
        return obs
    
    def _apply_rule_fusion(
        self,
        action: int,
        observation: np.ndarray,
        rule_flags: Dict[str, Any]
    ) -> Tuple[int, bool, Optional[str]]:
        """Apply rule-based action fusion."""
        overridden = False
        triggered_rule = None
        
        try:
            # Check cancellation ratio rule (example)
            if hasattr(self.rule_module, 'check_cancellation_ratio'):
                # Create dummy stats for rule checking
                stats = {'cancellation_ratio': 0.8}  # Placeholder
                
                if self.rule_module.check_cancellation_ratio(stats):
                    rule_flags['high_cancellation'] = True
                    
                    # Override action if fusion is enabled
                    fusion_cfg = self.cfg.get('fusion', {})
                    if fusion_cfg.get('rule_overrides', False):
                        action = 1  # Signal anomaly
                        overridden = True
                        triggered_rule = 'high_cancellation'
                        logger.debug(f"Action overridden by rule: {triggered_rule}")
            
        except Exception as e:
            logger.warning(f"Rule fusion error: {e}")
        
        return action, overridden, triggered_rule
    
    def _calculate_reward(self, action: int) -> Dict[str, float]:
        """Calculate reward components."""
        reward_components = {
            'detection_reward': 0.0,
            'trading_reward': 0.0,
            'risk_penalty': 0.0
        }
        
        # Detection reward (if targets available)
        if self.targets is not None and self.current_sequence_idx < len(self.targets):
            target = self.targets[self.current_sequence_idx]
            
            if action == 1:  # Signal anomaly
                if target > 0.001:  # Positive return threshold for "anomaly"
                    reward_components['detection_reward'] = self.detection_reward_scale
                else:
                    reward_components['detection_reward'] = -0.5 * self.detection_reward_scale
            elif target > 0.001 and action == 0:  # Missed anomaly
                reward_components['detection_reward'] = -self.detection_reward_scale
        
        # Trading reward (simple P&L proxy)
        if action == 2:  # Signal trade
            if self.current_sequence_idx < self.n_sequences - 1:
                # Get price change
                current_price = self._get_mid_price(self.current_sequence_idx)
                next_price = self._get_mid_price(self.current_sequence_idx + 1)
                
                if current_price > 0 and next_price > 0:
                    price_change = (next_price - current_price) / current_price
                    
                    # Simple long position
                    self.position = 1
                    reward_components['trading_reward'] = (
                        price_change * self.position * self.trading_reward_scale
                    )
        
        # Risk penalty (simple implementation)
        if self.episode_reward < -0.1:  # Arbitrary threshold
            reward_components['risk_penalty'] = -self.risk_penalty_scale
        
        return reward_components
    
    def _get_mid_price(self, sequence_idx: int) -> float:
        """Extract mid price from sequence (assumes first feature is mid_price_z)."""
        if sequence_idx >= self.n_sequences:
            return 0.0
        
        # Take last timestep of sequence, first feature (normalized mid price)
        # This is a simplified extraction - in practice you'd denormalize
        return float(self.sequences[sequence_idx, -1, 0])
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """Render environment state."""
        info = (
            f"MarketEnv - Step: {self.current_step}/{self.episode_length}, "
            f"Sequence: {self.current_sequence_idx}/{self.n_sequences}, "
            f"Episode Reward: {self.episode_reward:.4f}, "
            f"Position: {self.position}"
        )
        
        if mode == 'human':
            print(info)
        else:
            return info
    
    def close(self):
        """Clean up environment."""
        pass


if __name__ == "__main__":
    # Test the environment
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    try:
        from src.data.sequence_builder import load_sequences
        
        # Load sample sequences
        seq_path = "data/processed/sequences/sample_sequences.npz"
        if Path(seq_path).exists():
            sequences, targets, metadata = load_sequences(seq_path)
            
            print(f"Loaded sequences: {sequences.shape}")
            
            # Create environment
            cfg = {
                'episode_length': 10,
                'flatten_obs': False,
                'rewards': {
                    'detection_scale': 1.0,
                    'trading_scale': 0.1,
                    'risk_penalty_scale': 0.5
                }
            }
            
            env = MarketEnv(sequences, targets, cfg)
            
            print(f"Action space: {env.action_space}")
            print(f"Observation space: {env.observation_space}")
            
            # Run random policy
            obs = env.reset()
            total_reward = 0
            
            for step in range(10):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                total_reward += reward
                
                print(f"Step {step}: action={action}, reward={reward:.4f}, done={done}")
                print(f"  Info: {info['raw_reward_components']}")
                
                if done:
                    break
            
            print(f"Total reward: {total_reward:.4f}")
            env.render()
            
        else:
            print(f"Sample sequences not found at {seq_path}")
            print("Run sequence building first: python scripts/build_sequences.py ...")
            
    except Exception as e:
        print(f"Error testing environment: {e}")
        import traceback
        traceback.print_exc()