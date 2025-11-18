"""Policy utilities including rule-based fusion hooks."""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def fuse_action_with_rules(
    action: int,
    observation: np.ndarray,
    rule_module: Any,
    cfg: Dict[str, Any]
) -> Tuple[int, bool, Optional[str]]:
    """
    Apply rule-based fusion to RL actions.
    
    Args:
        action: Original RL action
        observation: Current observation
        rule_module: Module containing rule checking functions
        cfg: Configuration dictionary
    
    Returns:
        Tuple of (final_action, overridden_flag, triggered_rule_name)
    """
    overridden = False
    triggered_rule = None
    final_action = action
    
    # Get fusion configuration
    fusion_cfg = cfg.get('fusion', {})
    if not fusion_cfg.get('rule_overrides', False):
        return final_action, overridden, triggered_rule
    
    try:
        # Check various rules and apply overrides
        
        # Rule 1: High cancellation ratio
        if hasattr(rule_module, 'check_cancellation_ratio'):
            # Extract or simulate market statistics from observation
            stats = _extract_market_stats(observation, cfg)
            
            if rule_module.check_cancellation_ratio(stats):
                logger.debug("High cancellation ratio detected")
                
                # Override action based on rule configuration
                override_action = fusion_cfg.get('cancellation_override_action', 1)  # Signal anomaly
                if action != override_action:
                    final_action = override_action
                    overridden = True
                    triggered_rule = 'high_cancellation_ratio'
        
        # Rule 2: Extreme price movements (example)
        if hasattr(rule_module, 'check_extreme_movement') and not overridden:
            # Check for extreme price movements in observation
            if _check_extreme_price_movement(observation, fusion_cfg):
                logger.debug("Extreme price movement detected")
                
                override_action = fusion_cfg.get('extreme_movement_override_action', 1)
                if action != override_action:
                    final_action = override_action
                    overridden = True
                    triggered_rule = 'extreme_price_movement'
        
        # Rule 3: Risk threshold breach
        if hasattr(rule_module, 'check_risk_threshold') and not overridden:
            risk_stats = _extract_risk_stats(observation, cfg)
            
            if hasattr(rule_module, 'check_risk_threshold') and rule_module.check_risk_threshold(risk_stats):
                logger.debug("Risk threshold breached")
                
                # Force hold action when risk is too high
                override_action = 0  # Hold
                if action != override_action:
                    final_action = override_action
                    overridden = True
                    triggered_rule = 'risk_threshold_breach'
        
        if overridden:
            logger.info(f"Action overridden: {action} -> {final_action} (rule: {triggered_rule})")
    
    except Exception as e:
        logger.warning(f"Error in rule fusion: {e}")
    
    return final_action, overridden, triggered_rule


def _extract_market_stats(observation: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, float]:
    """Extract market statistics from observation for rule checking."""
    # This is a simplified implementation
    # In practice, you'd extract meaningful statistics from the observation
    
    stats = {
        'cancellation_ratio': 0.75,  # Placeholder - would compute from observation
        'volume_imbalance': 0.1,
        'price_volatility': 0.02
    }
    
    # If observation contains specific features, extract them
    if len(observation.shape) == 2:  # (seq_len, n_features)
        # Example: assume last timestep contains current market state
        current_state = observation[-1, :]
        
        # Extract features based on known feature order
        # This would depend on your specific feature engineering
        if len(current_state) >= 3:
            stats['cancellation_ratio'] = min(max(abs(current_state[2]), 0.0), 1.0)
    
    return stats


def _extract_risk_stats(observation: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, float]:
    """Extract risk-related statistics from observation."""
    risk_stats = {
        'volatility': 0.02,
        'drawdown': 0.05,
        'var_estimate': -0.1
    }
    
    # Compute rolling volatility from observation if possible
    if len(observation.shape) == 2 and observation.shape[0] > 1:
        # Assume first feature is price-related
        price_series = observation[:, 0]
        returns = np.diff(price_series)
        if len(returns) > 0:
            risk_stats['volatility'] = np.std(returns)
    
    return risk_stats


def _check_extreme_price_movement(observation: np.ndarray, fusion_cfg: Dict[str, Any]) -> bool:
    """Check for extreme price movements in observation."""
    threshold = fusion_cfg.get('extreme_movement_threshold', 3.0)  # 3 standard deviations
    
    if len(observation.shape) == 2 and observation.shape[0] > 1:
        # Check if any feature shows extreme movement
        for feature_idx in range(observation.shape[1]):
            feature_series = observation[:, feature_idx]
            
            # Check if latest value is extreme compared to recent history
            if len(feature_series) >= 5:
                recent_mean = np.mean(feature_series[-5:-1])
                recent_std = np.std(feature_series[-5:-1])
                current_value = feature_series[-1]
                
                if recent_std > 0:
                    z_score = abs(current_value - recent_mean) / recent_std
                    if z_score > threshold:
                        return True
    
    return False


class FusionWrapper(gym.Wrapper):
    """
    Environment wrapper that applies rule-based action fusion.
    
    This wrapper intercepts actions before they reach the base environment
    and applies rule-based overrides when certain conditions are met.
    """
    
    def __init__(
        self,
        env: gym.Env,
        rule_module: Any,
        cfg: Dict[str, Any]
    ):
        """
        Initialize fusion wrapper.
        
        Args:
            env: Base environment to wrap
            rule_module: Module containing rule checking functions
            cfg: Configuration dictionary
        """
        super().__init__(env)
        self.rule_module = rule_module
        self.cfg = cfg
        
        # Statistics tracking
        self.total_overrides = 0
        self.override_counts = {}
        
        logger.info("FusionWrapper initialized with rule-based action fusion")
    
    def step(self, action):
        """Execute step with rule-based action fusion."""
        # Get current observation for rule checking
        # Note: This is a simplified approach - in practice you might want
        # to store the observation from the previous step
        obs = self._get_current_observation()
        
        # Apply rule fusion
        original_action = action
        action, overridden, triggered_rule = fuse_action_with_rules(
            action, obs, self.rule_module, self.cfg
        )
        
        # Track override statistics
        if overridden:
            self.total_overrides += 1
            self.override_counts[triggered_rule] = self.override_counts.get(triggered_rule, 0) + 1
        
        # Execute step with potentially modified action
        obs, reward, done, info = self.env.step(action)
        
        # Add fusion information to info
        fusion_info = {
            'original_action': original_action,
            'final_action': action,
            'action_overridden': overridden,
            'triggered_rule': triggered_rule,
            'total_overrides': self.total_overrides
        }
        
        if isinstance(info, dict):
            info.update(fusion_info)
        else:
            info = fusion_info
        
        return obs, reward, done, info
    
    def _get_current_observation(self):
        """Get current observation for rule checking."""
        # This is a placeholder - in practice you'd need to maintain
        # the current observation state or get it from the environment
        if hasattr(self.env, '_get_observation'):
            return self.env._get_observation()
        else:
            # Return dummy observation
            return np.zeros(self.observation_space.shape)
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get fusion statistics."""
        return {
            'total_overrides': self.total_overrides,
            'override_counts': self.override_counts.copy(),
            'override_rate': self.total_overrides / max(getattr(self, '_step_count', 1), 1)
        }
    
    def reset_fusion_statistics(self):
        """Reset fusion statistics."""
        self.total_overrides = 0
        self.override_counts = {}


def wrap_env_with_fusion(
    env: gym.Env,
    rule_module: Any,
    cfg: Dict[str, Any]
) -> gym.Env:
    """
    Wrap environment with rule-based fusion if enabled in config.
    
    Args:
        env: Base environment
        rule_module: Module containing rule functions
        cfg: Configuration dictionary
    
    Returns:
        Environment with or without fusion wrapper
    """
    fusion_cfg = cfg.get('fusion', {})
    
    if fusion_cfg.get('enabled', False) and rule_module is not None:
        env = FusionWrapper(env, rule_module, cfg)
        logger.info("Applied fusion wrapper to environment")
    
    return env


# Example rule module for testing
class DummyRuleModule:
    """Dummy rule module for testing fusion functionality."""
    
    @staticmethod
    def check_cancellation_ratio(stats: Dict[str, float], threshold: float = 0.7) -> bool:
        """Check if cancellation ratio exceeds threshold."""
        return stats.get('cancellation_ratio', 0.0) > threshold
    
    @staticmethod
    def check_risk_threshold(stats: Dict[str, float], threshold: float = 0.1) -> bool:
        """Check if risk metrics exceed threshold."""
        return stats.get('volatility', 0.0) > threshold
    
    @staticmethod
    def check_extreme_movement(stats: Dict[str, float], threshold: float = 2.0) -> bool:
        """Check for extreme price movements."""
        return abs(stats.get('price_change', 0.0)) > threshold


if __name__ == "__main__":
    # Test fusion functionality
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
            cfg = {
                'episode_length': 20,
                'fusion': {
                    'enabled': True,
                    'rule_overrides': True,
                    'cancellation_override_action': 1,
                    'extreme_movement_threshold': 2.0
                }
            }
            
            base_env = MarketEnv(sequences, targets, cfg)
            
            # Create dummy rule module
            rule_module = DummyRuleModule()
            
            # Wrap with fusion
            env = wrap_env_with_fusion(base_env, rule_module, cfg)
            
            print("Testing fusion wrapper...")
            
            # Run test episode
            obs = env.reset()
            total_overrides = 0
            
            for step in range(15):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                
                if info.get('action_overridden', False):
                    total_overrides += 1
                    print(f"Step {step}: Action overridden by rule '{info['triggered_rule']}'")
                    print(f"  Original: {info['original_action']} -> Final: {info['final_action']}")
                
                if done:
                    break
            
            # Print fusion statistics
            if hasattr(env, 'get_fusion_statistics'):
                stats = env.get_fusion_statistics()
                print(f"\nFusion Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            
            print(f"Total overrides in test: {total_overrides}")
            
        else:
            print(f"Sample sequences not found at {seq_path}")
            
    except Exception as e:
        print(f"Error testing fusion: {e}")
        import traceback
        traceback.print_exc()