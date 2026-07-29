"""
moe-orbit-prefetch — full Object A source: orbit predictor + sparse MoE runtime.
"""

from .emergent_metrics import emerges_greater, local_mean_threshold
from .orbit_predictor import OrbitPredictor

__all__ = [
    "OrbitPredictor",
    "emerges_greater",
    "local_mean_threshold",
    "DynamicExpertStore",
    "SparseDeepseekRuntime",
    "MID",
    "ask",
    "chat_generate",
    "get_engine",
]

__version__ = "0.3.0"


def __getattr__(name: str):
    if name == "DynamicExpertStore":
        from .dynamic_expert_store import DynamicExpertStore

        return DynamicExpertStore
    if name in ("SparseDeepseekRuntime", "MID"):
        from . import sparse_moe_runtime as _rt

        return getattr(_rt, name)
    if name in ("ask", "chat_generate", "get_engine"):
        from . import deepseek_chat_engine as _chat

        return getattr(_chat, name)
    raise AttributeError(name)
