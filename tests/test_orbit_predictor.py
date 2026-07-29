#!/usr/bin/env python3
"""Minimal unit test: OrbitPredictor API (no weights)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moe_orbit_prefetch import OrbitPredictor


def test_predict_deposit_shapes():
    p = OrbitPredictor(n_experts=16, top_k=4, window=8)
    h = np.ones(32, dtype=np.float64)
    out = p.predict(h, tok_id=1)
    assert len(out) == 4
    assert all(0 <= e < 16 for e in out)
    p.deposit(h, [0, 1, 2, 3], tok_id=1)
    st = p.learning_stats()
    assert st["n_deposits"] == 1.0


if __name__ == "__main__":
    test_predict_deposit_shapes()
    print("OK")
