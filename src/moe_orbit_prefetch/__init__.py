"""
moe-orbit-prefetch — Object A: dynamic MoE expert residency via orbit from embedding h.
"""

from .emergent_metrics import emerges_greater, local_mean_threshold
from .orbit_predictor import OrbitPredictor

__all__ = [
    "OrbitPredictor",
    "emerges_greater",
    "local_mean_threshold",
    "DynamicExpertStore",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "DynamicExpertStore":
        from .dynamic_expert_store import DynamicExpertStore

        return DynamicExpertStore
    raise AttributeError(name)
