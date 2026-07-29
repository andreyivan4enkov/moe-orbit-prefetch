"""Unit tests: expert store concurrency / eviction / metrics (no HF weights)."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from moe_orbit_prefetch.dynamic_expert_store import DynamicExpertStore
from moe_orbit_prefetch.emergent_metrics import emerges_greater


def _toy_store() -> DynamicExpertStore:
    # Minimal fake index — get_expert will be monkeypatched in concurrency test
    return DynamicExpertStore(
        model_id="toy",
        index_path=Path("."),
        weight_map={},
    )


def test_drop_expert_and_cold_first_evict():
    st = _toy_store()
    st.hot[(1, 0)] = {"w": torch.zeros(2)}
    st.hot[(1, 1)] = {"w": torch.zeros(2)}
    st.s_env[(1, 0)] = 0.0  # cold
    st.s_env[(1, 1)] = 10.0
    dropped = st.evict_below_mean()
    assert (1, 0) in dropped
    assert (1, 0) not in st.hot
    assert (1, 1) in st.hot
    assert st.drop_expert(1, 1) is True
    assert st.n_hot() == 0


def test_cuda_direct_refused_by_default():
    st = _toy_store()
    st.weight_map["model.layers.1.mlp.experts.0.down_proj.weight"] = "x.safetensors"
    with pytest.raises(RuntimeError, match="allow_cuda_direct"):
        st.get_expert(1, 0, device="cuda")


def test_coalesced_concurrent_get_expert(monkeypatch):
    st = _toy_store()
    st.weight_map = {
        "model.layers.1.mlp.experts.0.down_proj.weight": "shard.safetensors",
        "model.layers.1.mlp.experts.0.gate_proj.weight": "shard.safetensors",
        "model.layers.1.mlp.experts.0.up_proj.weight": "shard.safetensors",
    }

    def fake_resolve(_shard: str):
        return Path("/tmp/does_not_need_to_exist_for_mock")

    class FakeF:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def keys(self):
            return list(st.weight_map.keys())

        def get_tensor(self, name: str):
            time.sleep(0.08)
            return torch.ones(4)

    def fake_open(*_a, **_k):
        return FakeF()

    monkeypatch.setattr(st, "resolve_shard_path", fake_resolve)
    monkeypatch.setattr(
        "moe_orbit_prefetch.dynamic_expert_store.safe_open", fake_open
    )

    started = threading.Event()
    out: list = []
    err: list = []

    def worker():
        started.wait(timeout=5)
        try:
            out.append(st.get_expert(1, 0, device="cpu"))
        except Exception as e:
            err.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    started.set()
    for t in threads:
        t.join(timeout=10)
    assert not err, err
    assert len(out) == 4
    assert st.n_loads == 1
    assert st.n_coalesced_waits >= 1
    assert st.n_hot() == 1


def test_emerges_greater_skips_nan():
    ok, st = emerges_greater([1.0, float("nan"), 2.0], [0.0, 0.0, 0.0])
    assert ok
    assert st["wins"] == 2
    assert st["skipped_nonfinite"] == 1


def test_classic_prefetch_predictors_api():
    from moe_orbit_prefetch.classic_prefetch_predictors import (
        FrequencyPredictor,
        LruPredictor,
        OnlineSgdPredictor,
        PrevCopyPredictor,
    )

    h = np.zeros(8)
    for Cls in (FrequencyPredictor, LruPredictor, PrevCopyPredictor, OnlineSgdPredictor):
        p = Cls(n_experts=64, top_k=6)
        pred = p.predict(h, tok_id=3)
        assert len(pred) == 6
        p.deposit(h, [1, 2, 3, 4, 5, 6], tok_id=3)
        hz = p.predict_token_horizon(h, 3, 2)
        assert len(hz) == 2
        assert p.learning_stats()["n_deposits"] == 1.0
