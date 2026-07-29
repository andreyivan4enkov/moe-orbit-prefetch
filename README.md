# moe-orbit-prefetch

**Object A — complete open source** for dynamic MoE expert residency  
(orbit from embedding/residual \(h\) → prefetch → deposit → sleep).

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/github/v/release/andreyivan4enkov/moe-orbit-prefetch)](https://github.com/andreyivan4enkov/moe-orbit-prefetch/releases)

**Fork it. Edit it. Take methods you need.**  
All first-party **source + math** live in this tree (see [docs/SOURCE_MANIFEST.md](docs/SOURCE_MANIFEST.md) and [docs/MATH.md](docs/MATH.md)).

| Doc | Why |
|---|---|
| [docs/MATH.md](docs/MATH.md) | **Full equations** (predict / deposit / sleep / emergent gate) |
| [docs/SOURCE_MANIFEST.md](docs/SOURCE_MANIFEST.md) | Every source file mapped |
| [docs/DESIGN_DYNAMIC_WEIGHTS.md](docs/DESIGN_DYNAMIC_WEIGHTS.md) | L0/L1/L2 design |
| [docs/EMBEDDING_PROTOCOL.md](docs/EMBEDDING_PROTOCOL.md) | Embedding channel protocol |
| [ATTRIBUTION.md](ATTRIBUTION.md) | Small use vs credit in larger systems |
| [WHAT_WE_CLAIM.md](WHAT_WE_CLAIM.md) | Honest claim boundary |
| [RELATED_WORK.md](RELATED_WORK.md) | Prior art |
| [README.ru.md](README.ru.md) | Русская витрина |

---

## License (open + attribution)

- **Apache License 2.0** — free to use, modify, redistribute. **No fee.**
- Keep `LICENSE` + `NOTICE` on redistribution (standard Apache).
- **Small experiments / snippets:** that is enough.
- **Substantial products / systems** that ship this method: please credit the upstream  
  (`moe-orbit-prefetch` / andreyivan4enkov) — details in [ATTRIBUTION.md](ATTRIBUTION.md).

Model **weights** are not in git (DeepSeek/HF terms). Everything else is.

---

## Install

```bash
git clone https://github.com/andreyivan4enkov/moe-orbit-prefetch.git
cd moe-orbit-prefetch
python -m venv .venv && source .venv/bin/activate
pip install -e ".[runtime]"
```

---

## Source you can edit

```text
src/moe_orbit_prefetch/
  orbit_predictor.py      # math: MATH.md §1–5
  emergent_metrics.py     # math: MATH.md §2,4,6
  dynamic_expert_store.py # math: MATH.md §7
  sparse_moe_runtime.py   # full sparse generate + prefetch
  deepseek_chat_engine.py
  process_gate.py
```

---

## Run

```bash
python examples/01_toy_orbit_no_weights.py
python examples/02_smoke_expert_slice.py          # needs HF cache
python examples/03_smoke_dynamic_weights_v13.py
python examples/04_chat_ask.py "Hello" --max-new 32
python examples/bench_humaneval_lean/bench_humaneval_lean.py   # long
```

---

## Verified lab (honest)

| Result | Meaning |
|---|---|
| [v13](results/v13_dynamic_expert_slice.md) | Expert slice + sleep ↓ RAM |
| [v36](results/v36_humaneval_prefetch_edge.md) | Code tie; miss-wait ours better; late learning↑ false |

We do **not** claim SOTA or identity with DeepSeek’s trained gate.
