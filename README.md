# moe-orbit-prefetch

**Object A — full open source: dynamic MoE expert residency + sparse DeepSeek runtime with orbit prefetch.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/github/v/release/andreyivan4enkov/moe-orbit-prefetch)](https://github.com/andreyivan4enkov/moe-orbit-prefetch/releases)

This repository is **fully open** under **MIT**. You get the **complete Object A source**: predictor, expert store, **full sparse generate runtime**, chat helper, smokes, and the lean HumanEval/QuixBugs bench script we actually ran.

**Model weights are not redistributed** (Hugging Face / DeepSeek license). Everything else you need to run is in this tree.

| Document | Purpose |
|---|---|
| [WHAT_WE_CLAIM.md](WHAT_WE_CLAIM.md) | Claim boundary |
| [RELATED_WORK.md](RELATED_WORK.md) | Prior art |
| [NOTICE.md](NOTICE.md) | Third-party / model notices |
| [docs/DESIGN_DYNAMIC_WEIGHTS.md](docs/DESIGN_DYNAMIC_WEIGHTS.md) | Design notes |
| [docs/TAGS.md](docs/TAGS.md) | GitHub topics |
| [results/](results/) | Verified lab artifacts |
| [README.ru.md](README.ru.md) | Русская витрина |

Object B (chat topology / RLM memory) is **out of scope**.

---

## What is in the source tree

```text
src/moe_orbit_prefetch/
  orbit_predictor.py        # S_env orbit from h (prefetch predictor)
  emergent_metrics.py       # local stats / wins>losses helpers
  dynamic_expert_store.py   # safetensors expert slice load + sleep
  process_gate.py           # wait if another heavy job is running
  sparse_moe_runtime.py     # FULL sparse DeepSeek-V2-Lite generate + modeled prefetch
  deepseek_chat_engine.py   # chat_generate / ask wrapper
examples/
  01_toy_orbit_no_weights.py
  02_smoke_expert_slice.py
  03_smoke_dynamic_weights_v13.py
  04_chat_ask.py
  bench_humaneval_lean/     # full lean HumanEval/QuixBugs A/B script
results/                    # sanitized reports + raw JSON from lab runs
```

---

## Install

```bash
git clone https://github.com/andreyivan4enkov/moe-orbit-prefetch.git
cd moe-orbit-prefetch
python -m venv .venv && source .venv/bin/activate
pip install -e ".[runtime]"
```

Optional for the long bench: `pip install -e ".[bench]"` (adds `psutil`).

---

## Run (complete usage path)

### Always works (no weights)

```bash
python examples/01_toy_orbit_no_weights.py
python tests/test_orbit_predictor.py
```

### Expert slice + sleep (needs HF cache)

```bash
huggingface-cli download deepseek-ai/DeepSeek-V2-Lite-Chat model.safetensors.index.json
# and the shard containing layer-1 experts (large)
python examples/02_smoke_expert_slice.py
python examples/03_smoke_dynamic_weights_v13.py
```

### Full chat via sparse runtime + orbit

```bash
python examples/04_chat_ask.py "Say hello in one short sentence." --max-new 32
```

First call loads the spine (~2.6GB fp16) and experts on demand. **Slow on CPU / 16GB laptop — expected.**

### Full lean code bench (hours on laptop)

```bash
pip install -e ".[bench]"
python examples/bench_humaneval_lean/bench_humaneval_lean.py
```

Lab outcome from our run: see [results/v36_humaneval_prefetch_edge.md](results/v36_humaneval_prefetch_edge.md) and JSON next to it.

---

## Python API

```python
from moe_orbit_prefetch import OrbitPredictor, DynamicExpertStore
from moe_orbit_prefetch import SparseDeepseekRuntime, ask

# predictor only
pred = OrbitPredictor(n_experts=64, top_k=6)

# full generate (weights from HF cache)
text = ask("Hello", max_new_tokens=32)

# or explicit runtime
rt = SparseDeepseekRuntime.load(use_modeled_prefetch=True, prefetch_horizon=2)
```

`OrbitPredictor` **does not replace** DeepSeek’s trained gate. It predicts which expert **weights** to prefetch; the live gate still chooses the true set at exec.

---

## Verified lab (honest)

| Artifact | Result |
|---|---|
| [v13](results/v13_dynamic_expert_slice.md) | Expert slice load + hit + sleep ↓ resident |
| [v36](results/v36_humaneval_prefetch_edge.md) | Code pass@1 **tie**; miss-wait **ours better**; learning↑ late **False** |

---

## License

**MIT** for all code and docs in this repository.  
Weights: upstream DeepSeek / Hugging Face terms only ([NOTICE.md](NOTICE.md)).
