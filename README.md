# moe-orbit-prefetch

**Object A — dynamic MoE expert residency from an orbit on embedding / residual `h`.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Plain idea: do **not** keep the entire MoE in RAM. Predict which **experts** will be needed from `h`, **prefetch** their weight slices from safetensors, **deposit** after the live gate’s true set, **sleep** cold experts.

This repository publishes **only Object A** (experts). Chat topological memory / RLM (Object B) is **out of scope**.

| Document | Purpose |
|---|---|
| [WHAT_WE_CLAIM.md](WHAT_WE_CLAIM.md) | Claim boundary (verified vs not) |
| [RELATED_WORK.md](RELATED_WORK.md) | Prior art |
| [docs/TAGS.md](docs/TAGS.md) | GitHub topics / release tags |
| [results/](results/) | Sanitized lab reports |
| [README.ru.md](README.ru.md) | Русская витрина |

---

## Verified (lab)

| Result | Meaning |
|---|---|
| [v13 expert slice](results/v13_dynamic_expert_slice.md) | Load one expert; hit; sleep reduced resident (~52MB→~17MB in that run) |
| [v36 prefetch edge](results/v36_humaneval_prefetch_edge.md) | Code pass@1 **tied** classic; **lower** expert miss-wait (wins 5 / losses 1); learning↑ late = **False** |

We do **not** claim SOTA, gate identity with DeepSeek, or Object B.

---

## Install

```bash
git clone https://github.com/andreyivan4enkov/moe-orbit-prefetch.git
cd moe-orbit-prefetch
python -m venv .venv && source .venv/bin/activate
pip install -e ".[examples]"
```

---

## Working examples

### 1) Always-on (no model weights)

```bash
python examples/01_toy_orbit_no_weights.py
```

Shows `OrbitPredictor.predict` / `deposit` on synthetic `h`.

### 2) Real expert slice (needs HF cache of DeepSeek-V2-Lite-Chat)

```bash
# weights are NOT in this repo — download under DeepSeek/HF license first
huggingface-cli download deepseek-ai/DeepSeek-V2-Lite-Chat model.safetensors.index.json
# plus the shard that contains layer-1 expert 0 (large)

python examples/02_smoke_expert_slice.py
```

If the shard is missing, the script **SKIPs** honestly (exit 0) instead of faking success.

---

## Core API

```python
from moe_orbit_prefetch import OrbitPredictor, DynamicExpertStore

pred = OrbitPredictor(n_experts=64, top_k=6)
experts = pred.predict(h)          # prefetch candidates
# ... load experts via DynamicExpertStore ...
pred.deposit(h, true_experts)      # true set from live MoE gate when available
```

`OrbitPredictor` **does not replace** DeepSeek’s trained router. It predicts residency for prefetch.

---

## Layout

```text
src/moe_orbit_prefetch/   # OrbitPredictor, DynamicExpertStore, metrics
examples/                 # runnable demos
results/                  # sanitized verified reports
docs/                     # tags, notes
```

---

## License

MIT for **this code**. Model weights remain under their upstream licenses ([LICENSE](LICENSE) NOTICE).
