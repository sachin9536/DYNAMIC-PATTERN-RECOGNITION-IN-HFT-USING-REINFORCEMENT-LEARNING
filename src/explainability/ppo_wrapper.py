"""Wrapper for PPO model to make it compatible with SHAP and LIME."""

import numpy as np
from pathlib import Path
from typing import Union, Optional
import warnings

try:
    from stable_baselines3 import PPO
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError as e:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print(f"Import warning: {e}")


class PPOExplainabilityWrapper:
    """
    Wrapper around PPO model for explainability with SHAP and LIME.
    
    This wrapper extracts scalar predictions from the PPO model that can be
    explained by SHAP and LIME. It uses the value function (V-value) as the
    scalar output, which represents the expected return from a given state.
    """
    
    def __init__(self, model_path: str, use_value_function: bool = True):
        """
        Initialize the PPO wrapper.
        
        Args:
            model_path: Path to the trained PPO model (.zip file)
            use_value_function: If True, use V-value; if False, use action probabilities
        """
        self.model_path = model_path
        self.use_value_function = use_value_function
        self.model = None
        self.policy = None
        self.observation_space = None
        self.action_space = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the PPO model from file."""
        try:
            model_path = Path(self.model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            logger.info(f"Loading PPO model from {self.model_path}")
            self.model = PPO.load(str(model_path))
            self.policy = self.model.policy
            self.observation_space = self.model.observation_space
            self.action_space = self.model.action_space
            
            logger.info(f"PPO model loaded successfully")
            logger.info(f"Observation space: {self.observation_space}")
            logger.info(f"Action space: {self.action_space}")
            
        except Exception as e:
            logger.error(f"Failed to load PPO model: {e}")
            raise
    
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict scalar values for observations.
        
        Args:
            observations: Array of observations, shape (n_samples, n_features) or (n_features,)
        
        Returns:
            Array of scalar predictions, shape (n_samples,)
        """
        try:
            # Ensure observations is 2D
            if observations.ndim == 1:
                observations = observations.reshape(1, -1)
            
            # Convert to float32
            observations = observations.astype(np.float32)
            
            if self.use_value_function:
                # Use value function (V-value) as scalar output
                values = self._predict_values(observations)
                return values
            else:
                # Use action probability or entropy as scalar output
                probs = self._predict_action_probabilities(observations)
                # Return probability of most likely action
                return np.max(probs, axis=1)
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return zeros as fallback
            n_samples = observations.shape[0] if observations.ndim > 1 else 1
            return np.zeros(n_samples)
    
    def _predict_values(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict value function (V-value) for observations.
        
        Args:
            observations: Array of observations, shape (n_samples, n_features)
        
        Returns:
            Array of V-values, shape (n_samples,)
        """
        try:
            import torch
            
            # Convert to tensor
            obs_tensor = torch.as_tensor(observations).float()
            
            # Get value predictions
            with torch.no_grad():
                self.policy.set_training_mode(False)
                
                # Extract features
                features = self.policy.extract_features(obs_tensor)
                
                # Get latent values
                if hasattr(self.policy, 'mlp_extractor'):
                    latent_vf = self.policy.mlp_extractor.forward_critic(features)
                else:
                    latent_vf = features
                
                # Get value predictions
                values = self.policy.value_net(latent_vf)
                
                # Convert to numpy
                values_np = values.cpu().numpy().flatten()
            
            logger.debug(f"Predicted values: {values_np}")
            return values_np
        
        except Exception as e:
            logger.error(f"Value prediction failed: {e}")
            return np.zeros(observations.shape[0])
    
    def _predict_action_probabilities(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict action probabilities for observations.
        
        Args:
            observations: Array of observations, shape (n_samples, n_features)
        
        Returns:
            Array of action probabilities, shape (n_samples, n_actions)
        """
        try:
            import torch
            import torch.nn.functional as F
            
            # Convert to tensor
            obs_tensor = torch.as_tensor(observations).float()
            
            # Get action distribution
            with torch.no_grad():
                self.policy.set_training_mode(False)
                
                # Extract features
                features = self.policy.extract_features(obs_tensor)
                
                # Get latent policy
                if hasattr(self.policy, 'mlp_extractor'):
                    latent_pi = self.policy.mlp_extractor.forward_actor(features)
                else:
                    latent_pi = features
                
                # Get action logits
                action_logits = self.policy.action_net(latent_pi)
                
                # Convert to probabilities
                action_probs = F.softmax(action_logits, dim=1)
                
                # Convert to numpy
                probs_np = action_probs.cpu().numpy()
            
            logger.debug(f"Predicted action probabilities: {probs_np}")
            return probs_np
        
        except Exception as e:
            logger.error(f"Action probability prediction failed: {e}")
            n_actions = self.action_space.n if hasattr(self.action_space, 'n') else 3
            return np.ones((observations.shape[0], n_actions)) / n_actions
    
    def predict_proba(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict probabilities (for compatibility with some explainability methods).
        
        Args:
            observations: Array of observations
        
        Returns:
            Array of probabilities, shape (n_samples, 2) for binary classification format
        """
        # Get scalar predictions
        values = self.predict(observations)
        
        # Convert to binary classification format (required by some LIME versions)
        # Normalize values to [0, 1] range
        values_normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)
        
        # Create probability matrix
        probs = np.column_stack([1 - values_normalized, values_normalized])
        
        return probs
    
    def __call__(self, observations: np.ndarray) -> np.ndarray:
        """Make the wrapper callable."""
        return self.predict(observations)
    
    def get_feature_names(self) -> list:
        """Get feature names based on observation space."""
        try:
            obs_shape = self.observation_space.shape
            
            if len(obs_shape) == 1:
                # Flattened observation
                n_features = obs_shape[0]
                return [f'feature_{i}' for i in range(n_features)]
            elif len(obs_shape) == 2:
                # Sequence observation (seq_len, n_features)
                seq_len, n_features = obs_shape
                # Flatten feature names
                feature_names = []
                for t in range(seq_len):
                    for f in range(n_features):
                        feature_names.append(f't{t}_f{f}')
                return feature_names
            else:
                # Unknown shape
                total_features = np.prod(obs_shape)
                return [f'feature_{i}' for i in range(total_features)]
        
        except Exception as e:
            logger.error(f"Failed to get feature names: {e}")
            return ['feature_0']


def load_ppo_for_explanation(
    model_path: str = "artifacts/final_model.zip",
    use_value_function: bool = True
) -> PPOExplainabilityWrapper:
    """
    Load a PPO model wrapped for explainability.
    
    Args:
        model_path: Path to the trained PPO model
        use_value_function: Whether to use value function (True) or action probs (False)
    
    Returns:
        PPOExplainabilityWrapper instance
    """
    return PPOExplainabilityWrapper(model_path, use_value_function)


if __name__ == "__main__":
    # Test the PPO wrapper
    try:
        print("Testing PPO Explainability Wrapper...")
        
        # Try to load a model
        model_paths = [
            "artifacts/final_model.zip",
            "artifacts/models/production_model.zip",
            "artifacts/synthetic/final_model.zip"
        ]
        
        wrapper = None
        for path in model_paths:
            try:
                print(f"\nTrying to load model from: {path}")
                wrapper = load_ppo_for_explanation(path)
                print(f"✅ Successfully loaded model from {path}")
                break
            except Exception as e:
                print(f"❌ Failed to load from {path}: {e}")
        
        if wrapper is None:
            print("\n❌ No model could be loaded. Please train a model first.")
        else:
            # Test prediction
            print("\n--- Testing Predictions ---")
            
            # Get observation shape
            obs_shape = wrapper.observation_space.shape
            print(f"Observation shape: {obs_shape}")
            
            # Create test observations
            if len(obs_shape) == 1:
                test_obs = np.random.randn(5, obs_shape[0]).astype(np.float32)
            else:
                # Flatten for testing
                flat_size = np.prod(obs_shape)
                test_obs = np.random.randn(5, flat_size).astype(np.float32)
            
            print(f"Test observations shape: {test_obs.shape}")
            
            # Test value prediction
            values = wrapper.predict(test_obs)
            print(f"Predicted values: {values}")
            print(f"Values shape: {values.shape}")
            
            # Test probability prediction
            probs = wrapper.predict_proba(test_obs)
            print(f"Predicted probabilities shape: {probs.shape}")
            print(f"Sample probabilities: {probs[0]}")
            
            # Test feature names
            feature_names = wrapper.get_feature_names()
            print(f"Number of features: {len(feature_names)}")
            print(f"First 5 feature names: {feature_names[:5]}")
            
            print("\n✅ PPO wrapper test completed successfully!")
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
