# moe-orbit-prefetch

**Object A — open MoE orbit prefetch** (math + runtime + **analysis artifacts**).

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/github/v/release/andreyivan4enkov/moe-orbit-prefetch)](https://github.com/andreyivan4enkov/moe-orbit-prefetch/releases)

If you only skimmed earlier tags: **start with analysis** — there are trajectories, plots, and gates you can open without downloading DeepSeek weights.

## Analyze first (no weights)

```bash
git clone https://github.com/andreyivan4enkov/moe-orbit-prefetch.git
cd moe-orbit-prefetch && pip install -e ".[analysis]"
python analysis/generate_orbit_trajectory.py
python analysis/plot_orbit_analysis.py
```

| Artifact | What it is |
|---|---|
| [analysis/REPORT.md](analysis/REPORT.md) | How to read the offline experiment |
| [analysis/data/orbit_trajectory_200.json](analysis/data/orbit_trajectory_200.json) | **~215KB** per-step JSON (200 steps) |
| [analysis/figures/hit_curves_vs_baselines.png](analysis/figures/hit_curves_vs_baselines.png) | Orbit vs prev/freq/cyclic |
| [analysis/figures/s_env_evolution.png](analysis/figures/s_env_evolution.png) | Field \(S_{\mathrm{env}}\) over time |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Mermaid system map |
| [docs/MATH.md](docs/MATH.md) | Full equations |
| [docs/SOURCE_MANIFEST.md](docs/SOURCE_MANIFEST.md) | Every editable source file |

On a learnable synthetic stream (not live DeepSeek gate): orbit mean hit **≈0.30** vs prev **≈0.17** / freq **≈0.14** / cyclic **≈0.08**; emergent gates vs those baselines **PASS** (wins>losses). Re-run the scripts to regenerate.

## Full source (edit / fork)

```text
src/moe_orbit_prefetch/
  orbit_predictor.py       # predictor math
  emergent_metrics.py      # local thresholds + wins>losses
  dynamic_expert_store.py  # safetensors expert slices + sleep
  sparse_moe_runtime.py    # ~900 lines sparse DeepSeek generate + prefetch
  deepseek_chat_engine.py  # ask / chat_generate
  process_gate.py
```

License: **Apache-2.0** (free). Substantial products: credit upstream — [ATTRIBUTION.md](ATTRIBUTION.md).  
Weights stay on Hugging Face (not redistributed).

## Runtime examples (needs HF cache)

```bash
pip install -e ".[runtime]"
python examples/02_smoke_expert_slice.py
python examples/04_chat_ask.py "Hello" --max-new 32
```

Lab (hardware-bound): [results/v36_humaneval_prefetch_edge.md](results/v36_humaneval_prefetch_edge.md).

## Docs index

| Doc | Purpose |
|---|---|
| [WHAT_WE_CLAIM.md](WHAT_WE_CLAIM.md) | Claim boundary |
| [RELATED_WORK.md](RELATED_WORK.md) | Prior art |
| [README.ru.md](README.ru.md) | Русская витрина |
| [analysis/README.md](analysis/README.md) | Analysis how-to |
