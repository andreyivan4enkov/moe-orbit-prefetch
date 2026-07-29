# Source manifest — complete Object A tree

This repository is meant to be **forked and edited**. Below is every first-party source file and what it contains. If something is missing from this list, open an issue.

## Package `src/moe_orbit_prefetch/` (edit these)

| File | Role | Math |
|---|---|---|
| `orbit_predictor.py` | Predict / deposit / prefetch orbit from \(h\) | [docs/MATH.md](MATH.md) §1–5 |
| `emergent_metrics.py` | Local thresholds + wins>losses gate | [docs/MATH.md](MATH.md) §2, §4, §6 |
| `dynamic_expert_store.py` | Safetensors expert slice load, hit, sleep | [docs/MATH.md](MATH.md) §7 |
| `sparse_moe_runtime.py` | Full sparse DeepSeek-V2-Lite generate + modeled prefetch | [docs/MATH.md](MATH.md) §8 |
| `deepseek_chat_engine.py` | `ask` / `chat_generate` wrapper | uses runtime |
| `process_gate.py` | Wait if another heavy Python job is running | ops |
| `__init__.py` | Public exports | — |

## Examples (run / copy / modify)

| Path | Purpose |
|---|---|
| `examples/01_toy_orbit_no_weights.py` | Orbit API without HF |
| `examples/02_smoke_expert_slice.py` | One expert + sleep |
| `examples/03_smoke_dynamic_weights_v13.py` | Full v13 residency smoke |
| `examples/04_chat_ask.py` | End-to-end chat |
| `examples/bench_humaneval_lean/bench_humaneval_lean.py` | Ours vs classic lean code bench |

## Docs

| Path | Purpose |
|---|---|
| `docs/MATH.md` | **All Object A equations** |
| `docs/DESIGN_DYNAMIC_WEIGHTS.md` | Design narrative (L0/L1/L2) |
| `docs/EMBEDDING_PROTOCOL.md` | MoE embedding channel protocol |
| `docs/TAGS.md` | GitHub topics |
| `ATTRIBUTION.md` | Small vs substantial use credit |
| `WHAT_WE_CLAIM.md` | Claim boundary |
| `RELATED_WORK.md` | Prior art |
| `NOTICE` | Apache NOTICE + model notice |

## Results

Sanitized lab artifacts under `results/` (markdown + JSON from real runs).

## Not in git (by license)

DeepSeek / HF **weight files** (`.safetensors`). Download separately.
