"""Rule-based explainability for market anomaly detection."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import json

try:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of rules for market anomaly detection."""
    THRESHOLD = "threshold"
    RANGE = "range"
    PATTERN = "pattern"
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"


@dataclass
class Rule:
    """Individual rule definition."""
    name: str
    rule_type: RuleType
    condition: Callable[[Dict[str, Any]], bool]
    explanation: str
    priority: int = 1
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MarketAnomalyRules:
    """Rule-based system for market anomaly detection and explanation."""
    
    def __init__(self):
        self.rules = []
        self.feature_thresholds = {
            'volume_spike': 3.0,
            'price_volatility': 2.5,
            'order_imbalance': 0.7,
            'spread_anomaly': 2.0,
            'momentum_shift': 2.0
        }
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default market anomaly rules."""
        
        # Volume spike rule
        self.add_rule(Rule(
            name="volume_spike",
            rule_type=RuleType.THRESHOLD,
            condition=lambda obs: self._check_volume_spike(obs),
            explanation="Trading volume significantly exceeds normal levels",
            priority=3,
            confidence=0.9,
            metadata={'threshold': self.feature_thresholds['volume_spike']}
        ))
        
        # Price volatility rule
        self.add_rule(Rule(
            name="high_volatility",
            rule_type=RuleType.THRESHOLD,
            condition=lambda obs: self._check_high_volatility(obs),
            explanation="Price volatility is abnormally high",
            priority=3,
            confidence=0.85,
            metadata={'threshold': self.feature_thresholds['price_volatility']}
        ))
        
        # Order imbalance rule
        self.add_rule(Rule(
            name="order_imbalance",
            rule_type=RuleType.THRESHOLD,
            condition=lambda obs: self._check_order_imbalance(obs),
            explanation="Significant imbalance between buy and sell orders",
            priority=2,
            confidence=0.8,
            metadata={'threshold': self.feature_thresholds['order_imbalance']}
        ))
        
        # Spread anomaly rule
        self.add_rule(Rule(
            name="spread_anomaly",
            rule_type=RuleType.THRESHOLD,
            condition=lambda obs: self._check_spread_anomaly(obs),
            explanation="Bid-ask spread is unusually wide",
            priority=2,
            confidence=0.75,
            metadata={'threshold': self.feature_thresholds['spread_anomaly']}
        ))
        
        # Momentum shift rule
        self.add_rule(Rule(
            name="momentum_shift",
            rule_type=RuleType.PATTERN,
            condition=lambda obs: self._check_momentum_shift(obs),
            explanation="Sudden change in price momentum detected",
            priority=2,
            confidence=0.7,
            metadata={'threshold': self.feature_thresholds['momentum_shift']}
        ))
        
        # Multi-factor anomaly rule
        self.add_rule(Rule(
            name="multi_factor_anomaly",
            rule_type=RuleType.PATTERN,
            condition=lambda obs: self._check_multi_factor_anomaly(obs),
            explanation="Multiple anomaly indicators triggered simultaneously",
            priority=4,
            confidence=0.95,
            metadata={'min_factors': 3}
        ))
        
        # Statistical outlier rule
        self.add_rule(Rule(
            name="statistical_outlier",
            rule_type=RuleType.STATISTICAL,
            condition=lambda obs: self._check_statistical_outlier(obs),
            explanation="Observation is a statistical outlier based on historical data",
            priority=1,
            confidence=0.6,
            metadata={'z_threshold': 3.0}
        ))
    
    def add_rule(self, rule: Rule):
        """Add a new rule to the system."""
        self.rules.append(rule)
        logger.debug(f"Added rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove a rule by name."""
        self.rules = [r for r in self.rules if r.name != rule_name]
        logger.debug(f"Removed rule: {rule_name}")
    
    def update_thresholds(self, thresholds: Dict[str, float]):
        """Update feature thresholds."""
        self.feature_thresholds.update(thresholds)
        logger.info(f"Updated thresholds: {thresholds}")
    
    def explain_observation(
        self,
        observation: Union[np.ndarray, Dict[str, Any]],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Explain an observation using rule-based logic.
        
        Args:
            observation: Input observation
            feature_names: Optional feature names
        
        Returns:
            Rule-based explanation
        """
        # Convert observation to dictionary format
        obs_dict = self._prepare_observation(observation, feature_names)
        
        # Evaluate all rules
        triggered_rules = []
        rule_details = {}
        
        for rule in self.rules:
            try:
                if rule.condition(obs_dict):
                    triggered_rules.append(rule.name)
                    rule_details[rule.name] = {
                        'type': rule.rule_type.value,
                        'explanation': rule.explanation,
                        'priority': rule.priority,
                        'confidence': rule.confidence,
                        'metadata': rule.metadata
                    }
            except Exception as e:
                logger.warning(f"Rule {rule.name} evaluation failed: {e}")
        
        # Calculate overall anomaly score
        anomaly_score = self._calculate_anomaly_score(triggered_rules, rule_details)
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(triggered_rules, rule_details)
        
        return {
            'method': 'rule',
            'triggered_rules': triggered_rules,
            'rule_details': rule_details,
            'anomaly_score': anomaly_score,
            'explanation_text': explanation_text,
            'observation_summary': self._summarize_observation(obs_dict),
            'total_rules_evaluated': len(self.rules)
        }
    
    def _prepare_observation(
        self,
        observation: Union[np.ndarray, Dict[str, Any]],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Convert observation to dictionary format."""
        if isinstance(observation, dict):
            return observation
        
        if isinstance(observation, np.ndarray):
            obs_flat = observation.flatten()
            if feature_names and len(feature_names) >= len(obs_flat):
                return dict(zip(feature_names, obs_flat))
            else:
                return {f'feature_{i}': val for i, val in enumerate(obs_flat)}
        
        # Fallback
        return {'observation': observation}
    
    def _check_volume_spike(self, obs: Dict[str, Any]) -> bool:
        """Check for volume spike anomaly."""
        volume_features = [k for k in obs.keys() if 'volume' in k.lower()]
        if not volume_features:
            return False
        
        for feature in volume_features:
            if abs(obs[feature]) > self.feature_thresholds['volume_spike']:
                return True
        return False
    
    def _check_high_volatility(self, obs: Dict[str, Any]) -> bool:
        """Check for high volatility anomaly."""
        volatility_features = [k for k in obs.keys() if 'volatility' in k.lower() or 'vol' in k.lower()]
        if not volatility_features:
            # Fallback: check for high variance in price-related features
            price_features = [k for k in obs.keys() if any(term in k.lower() for term in ['price', 'return', 'change'])]
            if price_features:
                values = [abs(obs[k]) for k in price_features]
                return max(values) > self.feature_thresholds['price_volatility']
        
        for feature in volatility_features:
            if abs(obs[feature]) > self.feature_thresholds['price_volatility']:
                return True
        return False
    
    def _check_order_imbalance(self, obs: Dict[str, Any]) -> bool:
        """Check for order imbalance anomaly."""
        imbalance_features = [k for k in obs.keys() if 'imbalance' in k.lower()]
        if not imbalance_features:
            # Look for buy/sell related features
            buy_features = [k for k in obs.keys() if 'buy' in k.lower()]
            sell_features = [k for k in obs.keys() if 'sell' in k.lower()]
            
            if buy_features and sell_features:
                buy_total = sum(obs[k] for k in buy_features)
                sell_total = sum(obs[k] for k in sell_features)
                total = abs(buy_total) + abs(sell_total)
                if total > 0:
                    imbalance = abs(buy_total - sell_total) / total
                    return imbalance > self.feature_thresholds['order_imbalance']
        
        for feature in imbalance_features:
            if abs(obs[feature]) > self.feature_thresholds['order_imbalance']:
                return True
        return False
    
    def _check_spread_anomaly(self, obs: Dict[str, Any]) -> bool:
        """Check for spread anomaly."""
        spread_features = [k for k in obs.keys() if 'spread' in k.lower()]
        if not spread_features:
            return False
        
        for feature in spread_features:
            if abs(obs[feature]) > self.feature_thresholds['spread_anomaly']:
                return True
        return False
    
    def _check_momentum_shift(self, obs: Dict[str, Any]) -> bool:
        """Check for momentum shift."""
        momentum_features = [k for k in obs.keys() if 'momentum' in k.lower()]
        if not momentum_features:
            # Look for trend or change features
            trend_features = [k for k in obs.keys() if any(term in k.lower() for term in ['trend', 'change', 'shift'])]
            momentum_features = trend_features
        
        for feature in momentum_features:
            if abs(obs[feature]) > self.feature_thresholds['momentum_shift']:
                return True
        return False
    
    def _check_multi_factor_anomaly(self, obs: Dict[str, Any]) -> bool:
        """Check if multiple anomaly factors are present."""
        factor_count = 0
        
        if self._check_volume_spike(obs):
            factor_count += 1
        if self._check_high_volatility(obs):
            factor_count += 1
        if self._check_order_imbalance(obs):
            factor_count += 1
        if self._check_spread_anomaly(obs):
            factor_count += 1
        if self._check_momentum_shift(obs):
            factor_count += 1
        
        return factor_count >= 3
    
    def _check_statistical_outlier(self, obs: Dict[str, Any]) -> bool:
        """Check for statistical outliers."""
        values = [v for v in obs.values() if isinstance(v, (int, float))]
        if not values:
            return False
        
        # Simple z-score based outlier detection
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return False
        
        z_scores = [(v - mean_val) / std_val for v in values]
        max_z = max(abs(z) for z in z_scores)
        
        return max_z > 3.0
    
    def _calculate_anomaly_score(
        self,
        triggered_rules: List[str],
        rule_details: Dict[str, Any]
    ) -> float:
        """Calculate overall anomaly score based on triggered rules."""
        if not triggered_rules:
            return 0.0
        
        # Weighted score based on priority and confidence
        total_score = 0.0
        max_possible_score = 0.0
        
        for rule_name in triggered_rules:
            details = rule_details[rule_name]
            priority = details['priority']
            confidence = details['confidence']
            
            rule_score = priority * confidence
            total_score += rule_score
            max_possible_score += priority  # Assuming max confidence of 1.0
        
        # Normalize to 0-1 range
        if max_possible_score > 0:
            normalized_score = min(total_score / max_possible_score, 1.0)
        else:
            normalized_score = 0.0
        
        return normalized_score
    
    def _generate_explanation_text(
        self,
        triggered_rules: List[str],
        rule_details: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation text."""
        if not triggered_rules:
            return "No anomalies detected based on rule-based analysis."
        
        # Sort rules by priority
        sorted_rules = sorted(
            triggered_rules,
            key=lambda r: rule_details[r]['priority'],
            reverse=True
        )
        
        explanations = []
        for rule_name in sorted_rules:
            details = rule_details[rule_name]
            confidence_text = f"(confidence: {details['confidence']:.1%})"
            explanations.append(f"• {details['explanation']} {confidence_text}")
        
        if len(explanations) == 1:
            return f"Anomaly detected: {explanations[0]}"
        else:
            return f"Multiple anomalies detected:\n" + "\n".join(explanations)
    
    def _summarize_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics for the observation."""
        numeric_values = [v for v in obs.values() if isinstance(v, (int, float))]
        
        if not numeric_values:
            return {'error': 'No numeric values found'}
        
        return {
            'n_features': len(obs),
            'n_numeric': len(numeric_values),
            'mean_value': float(np.mean(numeric_values)),
            'std_value': float(np.std(numeric_values)),
            'min_value': float(np.min(numeric_values)),
            'max_value': float(np.max(numeric_values)),
            'extreme_values': sum(1 for v in numeric_values if abs(v) > 2.0)
        }
    
    def get_rule_summary(self) -> Dict[str, Any]:
        """Get summary of all rules in the system."""
        rule_types = {}
        for rule in self.rules:
            rule_type = rule.rule_type.value
            if rule_type not in rule_types:
                rule_types[rule_type] = 0
            rule_types[rule_type] += 1
        
        return {
            'total_rules': len(self.rules),
            'rule_types': rule_types,
            'rule_names': [r.name for r in self.rules],
            'thresholds': self.feature_thresholds.copy()
        }


if __name__ == "__main__":
    # Test rule-based explainability
    try:
        print("Testing rule-based explainability...")
        
        # Initialize rule system
        rule_system = MarketAnomalyRules()
        
        # Test normal observation
        normal_obs = {
            'volume_change': 0.5,
            'price_volatility': 1.0,
            'order_imbalance': 0.3,
            'spread': 0.8,
            'momentum': 0.2
        }
        
        print("Testing normal observation...")
        result = rule_system.explain_observation(normal_obs)
        print(f"Triggered rules: {result['triggered_rules']}")
        print(f"Anomaly score: {result['anomaly_score']:.3f}")
        print(f"Explanation: {result['explanation_text']}")
        
        # Test anomalous observation
        anomalous_obs = {
            'volume_change': 4.0,  # High volume spike
            'price_volatility': 3.5,  # High volatility
            'order_imbalance': 0.8,  # High imbalance
            'spread': 2.5,  # Wide spread
            'momentum': 3.0  # Strong momentum shift
        }
        
        print("\nTesting anomalous observation...")
        result = rule_system.explain_observation(anomalous_obs)
        print(f"Triggered rules: {result['triggered_rules']}")
        print(f"Anomaly score: {result['anomaly_score']:.3f}")
        print(f"Explanation: {result['explanation_text']}")
        
        # Test with numpy array
        print("\nTesting with numpy array...")
        array_obs = np.array([4.0, 3.5, 0.8, 2.5, 3.0])
        feature_names = ['volume_change', 'price_volatility', 'order_imbalance', 'spread', 'momentum']
        result = rule_system.explain_observation(array_obs, feature_names)
        print(f"Triggered rules: {result['triggered_rules']}")
        print(f"Anomaly score: {result['anomaly_score']:.3f}")
        
        # Get rule summary
        print("\nRule system summary:")
        summary = rule_system.get_rule_summary()
        print(f"Total rules: {summary['total_rules']}")
        print(f"Rule types: {summary['rule_types']}")
        
        print("Rule-based explainability test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()