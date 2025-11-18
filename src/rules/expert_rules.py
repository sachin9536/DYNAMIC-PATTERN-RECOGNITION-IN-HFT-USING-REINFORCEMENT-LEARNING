"""Rule-based expert module (simple, testable rules)."""
from typing import Dict, Any


def check_cancellation_ratio(stats: Dict[str, Any], threshold: float = 0.7) -> bool:
    """Return True if cancellations ratio exceeds threshold."""
    return stats.get('cancellation_ratio', 0.0) > threshold