# Analysis guide

If the repository looked “empty”, start here. This folder is for **offline analysis**.

## Quick start (no DeepSeek weights)

```bash
cd moe-orbit-prefetch
pip install -e ".[analysis]"
python analysis/generate_orbit_trajectory.py
python analysis/plot_orbit_analysis.py
```

Outputs:

- `analysis/data/orbit_trajectory_200.json` — **200 steps**, full per-step records  
- `analysis/figures/hit_curves_vs_baselines.png`  
- `analysis/figures/s_env_evolution.png`  
- `analysis/figures/mean_hit_bars.png`  
- `analysis/figures/gates_summary.json`

Open the PNGs on GitHub or locally. Load the JSON in pandas / Julia / R.

## What each step record contains

`t`, `regime`, `tok_id`, `h_norm`, `h0`, `h1`, `true`, `orbit`, `prev_copy`, `frequency`, `cyclic`,
`hit_*`, `s_env_mean_pos`, `s_env_max`, `s_env_nnz`, `learn` (deposit stats).

## Classic baselines included

| Baseline | Rule |
|---|---|
| `orbit` | `OrbitPredictor` (this method) |
| `prev_copy` | copy previous true top-k |
| `frequency` | most frequent experts so far |
| `cyclic` | rotating ids |

Gates use `emerges_greater` (wins > losses), not magic thresholds.

## Real-weight analysis

```bash
pip install -e ".[runtime]"
python examples/02_smoke_expert_slice.py
python examples/03_smoke_dynamic_weights_v13.py
```

Then inspect resident bytes / hits printed by the scripts and lab files under `results/`.
