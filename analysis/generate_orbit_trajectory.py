#!/usr/bin/env python3
"""
Generate a rich, analyzable orbit trajectory (no model weights).

Synthetic ground truth = top-k experts by cosine to fixed expert prototypes in h-space
(learnable by OrbitPredictor's history resonance + deposit). This is for ANALYSIS of the
predictor field — not a claim about DeepSeek live-gate accuracy.

Baselines on the same stream: prev_copy, frequency, cyclic.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_orbit_prefetch import OrbitPredictor, emerges_greater

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_prototypes(rng: np.random.Generator, n_experts: int, d: int) -> np.ndarray:
    p = rng.normal(size=(n_experts, d))
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-12
    return p


def true_experts(h: np.ndarray, prototypes: np.ndarray, top_k: int) -> list[int]:
    hn = h / (np.linalg.norm(h) + 1e-12)
    scores = prototypes @ hn
    return [int(i) for i in np.argsort(-scores)[:top_k]]


def pred_prev_copy(prev: list[int] | None, top_k: int) -> list[int]:
    if not prev:
        return list(range(top_k))
    return list(prev)[:top_k]


def pred_frequency(counts: Counter, top_k: int) -> list[int]:
    if not counts:
        return list(range(top_k))
    return [int(e) for e, _ in counts.most_common(top_k)]


def pred_cyclic(t: int, n_experts: int, top_k: int) -> list[int]:
    return [(t + i) % n_experts for i in range(top_k)]


def hit_frac(pred: list[int], true: list[int]) -> float:
    return len(set(pred) & set(true)) / max(1, len(true))


def main() -> int:
    rng = np.random.default_rng(42)
    n_experts, top_k, steps, d = 64, 6, 200, 64
    prototypes = make_prototypes(rng, n_experts, d)
    pred = OrbitPredictor(n_experts=n_experts, top_k=top_k, window=48)

    steps_out: list[dict] = []
    hits_orbit: list[float] = []
    hits_prev: list[float] = []
    hits_freq: list[float] = []
    hits_cyc: list[float] = []
    counts: Counter = Counter()
    prev_true: list[int] | None = None

    # four regimes: drift the mean of h so active experts change
    centers = [
        np.array([1.0, 0.2] + [0.0] * (d - 2)),
        np.array([-1.0, 0.2] + [0.0] * (d - 2)),
        np.array([0.2, 1.0] + [0.0] * (d - 2)),
        np.array([0.2, -1.0] + [0.0] * (d - 2)),
    ]

    for t in range(steps):
        regime = t // 50
        mu = centers[regime % 4]
        h = rng.normal(loc=mu, scale=0.25, size=d)
        true = true_experts(h, prototypes, top_k)
        tok_id = int(t % 17)

        o = pred.predict(h, tok_id=tok_id)
        p = pred_prev_copy(prev_true, top_k)
        f = pred_frequency(counts, top_k)
        c = pred_cyclic(t, n_experts, top_k)

        ho = hit_frac(o, true)
        hp = hit_frac(p, true)
        hf = hit_frac(f, true)
        hc = hit_frac(c, true)
        hits_orbit.append(ho)
        hits_prev.append(hp)
        hits_freq.append(hf)
        hits_cyc.append(hc)

        s_pos = pred.s_env[pred.s_env > 0]
        steps_out.append(
            {
                "t": t,
                "regime": regime,
                "tok_id": tok_id,
                "h_norm": float(np.linalg.norm(h)),
                "h0": float(h[0]),
                "h1": float(h[1]),
                "true": true,
                "orbit": o,
                "prev_copy": p,
                "frequency": f,
                "cyclic": c,
                "hit_orbit": ho,
                "hit_prev": hp,
                "hit_freq": hf,
                "hit_cyclic": hc,
                "s_env_mean_pos": float(np.mean(s_pos)) if s_pos.size else 0.0,
                "s_env_max": float(pred.s_env.max()),
                "s_env_nnz": int(np.sum(pred.s_env > 0)),
                "learn": pred.learning_stats(),
            }
        )

        pred.deposit(h, true, tok_id=tok_id)
        for e in true:
            counts[e] += 1
        prev_true = true

    half = steps // 2
    gates = {}
    for label, a, b in (
        ("orbit_vs_prev", hits_orbit, hits_prev),
        ("orbit_vs_freq", hits_orbit, hits_freq),
        ("orbit_vs_cyclic", hits_orbit, hits_cyc),
        ("orbit_late_vs_early", hits_orbit[half:], hits_orbit[:half]),
    ):
        ok, st = emerges_greater(a, b)
        gates[label] = {
            "pass": ok,
            **st,
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
        }

    payload = {
        "meta": {
            "n_experts": n_experts,
            "top_k": top_k,
            "steps": steps,
            "d": d,
            "seed": 42,
            "ground_truth": "topk cosine(h, fixed expert prototypes)",
            "note": "Synthetic learnable stream for ANALYSIS of OrbitPredictor — not live DeepSeek gate.",
        },
        "gates": gates,
        "summary": {
            "mean_hit_orbit": float(np.mean(hits_orbit)),
            "mean_hit_prev": float(np.mean(hits_prev)),
            "mean_hit_freq": float(np.mean(hits_freq)),
            "mean_hit_cyclic": float(np.mean(hits_cyc)),
        },
        "steps": steps_out,
    }

    out = OUT_DIR / "orbit_trajectory_200.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print("summary:", json.dumps(payload["summary"], indent=2))
    for k, v in gates.items():
        print(
            f"  {k}: pass={v['pass']} wins={v['wins']} losses={v['losses']} "
            f"mean_a={v['mean_a']:.3f} mean_b={v['mean_b']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
