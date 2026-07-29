#!/usr/bin/env python3
"""
Sensitivity of OrbitPredictor under scaled cosine / resonance regimes.

Tier S only (no DeepSeek weights). Reports how mean prefetch-hit and field nnz
move when we stretch the similarity structure of h — diagnostic tables, not a
PASS/FAIL magic margin.

See docs/RISKS.md R2 and docs/EVIDENCE_TIERS.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moe_orbit_prefetch import OrbitPredictor, emerges_greater

OUT = Path(__file__).resolve().parent / "figures" / "sensitivity_resonance.json"


def unit(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x)) + 1e-12
    return x / n


def run_stream(
    rng: np.random.Generator,
    *,
    stretch: float,
    n_steps: int = 200,
    n_experts: int = 64,
    top_k: int = 6,
    d: int = 64,
) -> dict:
    """
    stretch > 1: pull h toward current regime prototype (stronger cosine structure).
    stretch < 1: flatten toward isotropic noise (weaker structure).
    """
    prototypes = rng.normal(size=(n_experts, d))
    prototypes = np.stack([unit(p) for p in prototypes])
    pred = OrbitPredictor(n_experts=n_experts, top_k=top_k, window=48)

    hits: list[float] = []
    nnz: list[int] = []

    for t in range(n_steps):
        regime = (t // 50) % 4
        center = prototypes[regime * (n_experts // 4)]
        noise = rng.normal(size=d)
        s = float(stretch)
        raw = s * center + (1.0 - min(1.0, s)) * noise + 0.15 * noise
        h = unit(raw.astype(np.float64))
        scores = prototypes @ h
        true = list(np.argsort(-scores)[:top_k])
        orbit = pred.predict(h, tok_id=None)
        hits.append(len(set(orbit) & set(true)) / float(top_k))
        pred.deposit(h, true, tok_id=None)
        nnz.append(int(np.count_nonzero(pred.s_env > 0)))

    early = float(np.mean(hits[:40])) if hits else 0.0
    late = float(np.mean(hits[-40:])) if hits else 0.0
    return {
        "stretch": stretch,
        "mean_hit": float(np.mean(hits)),
        "early_hit": early,
        "late_hit": late,
        "mean_nnz": float(np.mean(nnz)),
        "hits": hits,
    }


def main() -> int:
    base_rng = np.random.default_rng(0)
    rows_out = []
    hit_series: dict[float, list[float]] = {}
    for stretch in (0.25, 0.5, 1.0, 1.5, 2.0):
        # independent stream per stretch, same seed recipe offset
        rng = np.random.default_rng(int(1000 + stretch * 100))
        row = run_stream(rng, stretch=stretch)
        hit_series[stretch] = row.pop("hits")
        rows_out.append(row)

    # Emergent compare: stronger structure (2.0) vs weaker (0.25) hit series
    ok_strong, st_strong = emerges_greater(hit_series[2.0], hit_series[0.25])
    ok_mid, st_mid = emerges_greater(hit_series[1.0], hit_series[0.5])

    payload = {
        "tier": "S",
        "note": (
            "Sensitivity of OrbitPredictor on synthetic stream under h-structure stretch. "
            "Not a live DeepSeek gate study. Not a magic PASS threshold. "
            "Expect: weaker structure → usually lower hit (R3 risk visible here)."
        ),
        "rows": rows_out,
        "gates_diagnostic": {
            "stretch_2.0_vs_0.25": {"pass": ok_strong, **{k: st_strong[k] for k in ("wins", "losses", "gap")}},
            "stretch_1.0_vs_0.5": {"pass": ok_mid, **{k: st_mid[k] for k in ("wins", "losses", "gap")}},
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")
    _ = base_rng  # silence lint
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
