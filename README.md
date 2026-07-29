# moe-orbit-prefetch

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/andreyivan4enkov/moe-orbit-prefetch/actions/workflows/ci.yml/badge.svg)](https://github.com/andreyivan4enkov/moe-orbit-prefetch/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/github/v/release/andreyivan4enkov/moe-orbit-prefetch)](https://github.com/andreyivan4enkov/moe-orbit-prefetch/releases)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange.svg)](MODEL_CARD.md)

**Object A** — dynamic MoE **expert prefetch / residency** from embedding/residual `h`  
(sparse DeepSeek-V2-Lite path + math + lean lab evidence).

> **Status:** research prototype (alpha). Not a production inference server.  
> **Weights:** not redistributed (Hugging Face / DeepSeek terms).  
> **Lab compute:** author MacBook-class laptop — see [docs/LAB_SCOPE.md](docs/LAB_SCOPE.md).

| Start here | Link |
|---|---|
| What we claim | [WHAT_WE_CLAIM.md](WHAT_WE_CLAIM.md) |
| Model / method card | [MODEL_CARD.md](MODEL_CARD.md) |
| Live vs synthetic evidence | [docs/EVIDENCE_TIERS.md](docs/EVIDENCE_TIERS.md) |
| Lab hardware limits | [docs/LAB_SCOPE.md](docs/LAB_SCOPE.md) |
| Auditor issue triage | [docs/AUDITOR_ISSUES.md](docs/AUDITOR_ISSUES.md) |
| Русская витрина | [README.ru.md](README.ru.md) |

## Quick facts (honest)

| Question | Answer |
|---|---|
| Real DeepSeek tests? | **Yes** — Tier L: `results/v13_*`, `results/v36_*` |
| 100-task GPU HumanEval? | **Not claimed** — out of author laptop scope |
| Synthetic plots? | Tier **S** only — not live-gate proof |
| v36 code quality | pass@1 **tie** with classic |
| v36 prefetch | miss-wait **ours better** (wins>losses) |
| Late orbit learning↑ | **False** on that lean run |

## Install

```bash
git clone https://github.com/andreyivan4enkov/moe-orbit-prefetch.git
cd moe-orbit-prefetch
pip install -e ".[runtime]"   # needs torch / HF for live path
# or
pip install -e ".[analysis]"  # no weights; plots + unit-level scripts
```

## Tier L — real model (needs HF cache)

```bash
python examples/02_smoke_expert_slice.py
python examples/03_smoke_dynamic_weights_v13.py
python examples/04_chat_ask.py "Hello" --max-new 32
# lean bench (hours on CPU laptop):
python examples/bench_humaneval_lean/bench_humaneval_lean.py
```

Published snapshots: [results/v36_humaneval_prefetch_edge.md](results/v36_humaneval_prefetch_edge.md).

## Tier S — offline analysis (no weights)

```bash
python analysis/generate_orbit_trajectory.py
python analysis/plot_orbit_analysis.py
python analysis/sensitivity_resonance_decay.py
```

## Source layout

```text
src/moe_orbit_prefetch/
  orbit_predictor.py
  emergent_metrics.py
  dynamic_expert_store.py
  sparse_moe_runtime.py
  deepseek_chat_engine.py
  process_gate.py
docs/MATH.md              # equations
MODEL_CARD.md             # ML-style method card
```

## License & citation

- **Apache-2.0** — [LICENSE](LICENSE), [NOTICE](NOTICE)
- Substantial products: [ATTRIBUTION.md](ATTRIBUTION.md)
- Cite: [CITATION.cff](CITATION.cff)

```bibtex
@software{moe_orbit_prefetch,
  author = {Ivanchenkov, Andrey},
  title = {moe-orbit-prefetch},
  year = {2026},
  url = {https://github.com/andreyivan4enkov/moe-orbit-prefetch},
  license = {Apache-2.0},
  version = {0.5.0}
}
```

## Contributing / security / conduct

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [CHANGELOG.md](CHANGELOG.md)

## LLM auditors

Do **not** paste a ZIP as text (`PK` / Deflate). Prefer raw file URLs or:

```bash
python tools/export_plaintext_for_audit.py
```

Point reviewers at [docs/EVIDENCE_TIERS.md](docs/EVIDENCE_TIERS.md) and [docs/LAB_SCOPE.md](docs/LAB_SCOPE.md).
