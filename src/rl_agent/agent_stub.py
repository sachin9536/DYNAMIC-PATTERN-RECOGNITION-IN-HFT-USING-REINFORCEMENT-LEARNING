"""RL Agent interface/stub. Actual training code added later."""
from typing import Any, Dict


class RLAgent:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def act(self, obs):
        """Return an action given an observation. (stub)"""
        raise NotImplementedError

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass