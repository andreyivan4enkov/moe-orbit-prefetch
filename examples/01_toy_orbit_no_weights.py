#!/usr/bin/env python3
"""
Example 01 — OrbitPredictor without model weights (always runnable).

Demonstrates Object A law on synthetic (h, true_experts):
  predict → compare to truth → deposit → field evolves.

No Hugging Face download. No DeepSeek weights.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from moe_orbit_prefetch import OrbitPredictor, emerges_greater


def main() -> int:
    rng = np.random.default_rng(0)
    n_experts, top_k, steps = 64, 6, 40
    pred = OrbitPredictor(n_experts=n_experts, top_k=top_k, window=24)

    hits_early: list[float] = []
    hits_late: list[float] = []

    # synthetic: experts cluster by sign of h[0]
    for t in range(steps):
        h = rng.normal(size=128).astype(np.float64)
        base = 0 if h[0] >= 0 else 32
        true = [(base + i) % n_experts for i in range(top_k)]
        pred_ids = pred.predict(h, tok_id=t % 10)
        hit = len(set(pred_ids) & set(true)) / top_k
        (hits_early if t < steps // 2 else hits_late).append(hit)
        pred.deposit(h, true, tok_id=t % 10)

    better, st = emerges_greater(hits_late, hits_early)
    print("OrbitPredictor toy example (no weights)")
    print(f"  early mean hit={float(np.mean(hits_early)):.3f}")
    print(f"  late  mean hit={float(np.mean(hits_late)):.3f}")
    print(f"  late>early (emergent wins>losses): {better}  {st}")
    print(f"  learning_stats: {pred.learning_stats()}")
    # Toy must show the API works; late>early is typical but not a published claim gate.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
