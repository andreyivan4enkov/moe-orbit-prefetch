#!/usr/bin/env python3
"""
Plot analysis figures from orbit_trajectory_200.json.

Produces PNGs under analysis/figures/ for GitHub viewing / papers.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "orbit_trajectory_200.json"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def moving_avg(x: np.ndarray, w: int = 11) -> np.ndarray:
    if len(x) < w:
        return x
    ker = np.ones(w) / w
    return np.convolve(x, ker, mode="same")


def main() -> int:
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    steps = payload["steps"]
    t = np.array([s["t"] for s in steps])
    ho = np.array([s["hit_orbit"] for s in steps], dtype=float)
    hp = np.array([s["hit_prev"] for s in steps], dtype=float)
    hf = np.array([s["hit_freq"] for s in steps], dtype=float)
    hc = np.array([s["hit_cyclic"] for s in steps], dtype=float)
    smean = np.array([s["s_env_mean_pos"] for s in steps], dtype=float)
    snnz = np.array([s["s_env_nnz"] for s in steps], dtype=float)
    regimes = np.array([s["regime"] for s in steps])

    # Fig 1: hit curves
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, moving_avg(ho), label="orbit", linewidth=2)
    ax.plot(t, moving_avg(hp), label="prev_copy", alpha=0.85)
    ax.plot(t, moving_avg(hf), label="frequency", alpha=0.85)
    ax.plot(t, moving_avg(hc), label="cyclic", alpha=0.7)
    for r in range(int(regimes.max()) + 1):
        ax.axvline(r * 50, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("hit fraction (moving avg)")
    ax.set_title("Prefetch hit vs classical baselines (synthetic structured stream)")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    p1 = FIG / "hit_curves_vs_baselines.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)

    # Fig 2: field mass
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, smean, label="mean positive S_env")
    ax.set_xlabel("step")
    ax.set_ylabel("S_env mean (positive)")
    ax2 = ax.twinx()
    ax2.plot(t, snnz, color="orange", alpha=0.7, label="nnz(S_env>0)")
    ax.set_title("Stigmergy field evolution")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    p2 = FIG / "s_env_evolution.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)

    # Fig 3: bar summary
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["orbit", "prev", "freq", "cyclic"]
    means = [ho.mean(), hp.mean(), hf.mean(), hc.mean()]
    ax.bar(names, means, color=["#2a6fdb", "#888", "#888", "#888"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean hit")
    ax.set_title("Mean hit over 200 steps")
    for i, m in enumerate(means):
        ax.text(i, m + 0.02, f"{m:.3f}", ha="center")
    fig.tight_layout()
    p3 = FIG / "mean_hit_bars.png"
    fig.savefig(p3, dpi=140)
    plt.close(fig)

    # Fig 4: gate table as image-like text file companion
    gates_path = FIG / "gates_summary.json"
    gates_path.write_text(json.dumps(payload["gates"], indent=2), encoding="utf-8")

    print("wrote", p1)
    print("wrote", p2)
    print("wrote", p3)
    print("wrote", gates_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
