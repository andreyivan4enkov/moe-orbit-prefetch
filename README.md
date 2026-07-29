# moe-orbit-prefetch

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/andreyivan4enkov/moe-orbit-prefetch/actions/workflows/ci.yml/badge.svg)](https://github.com/andreyivan4enkov/moe-orbit-prefetch/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/github/v/release/andreyivan4enkov/moe-orbit-prefetch)](https://github.com/andreyivan4enkov/moe-orbit-prefetch/releases)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange.svg)](MODEL_CARD.md)

**Object A** — dynamic MoE **expert prefetch / residency** from embedding/residual `h`  
for **open DeepSeek-style MoE** (DeepSeek-V2-Lite **and** GigaChat lab paths).

> **Status:** research prototype (alpha). Not a production inference server.  
> **Weights:** not redistributed (Hugging Face / model vendor terms).  
> **Lab compute:** MacBook Pro (2019), Intel i9, 16 GB RAM, Radeon 4 GB; thermally constrained — see [docs/LAB_SCOPE.md](docs/LAB_SCOPE.md).  
> **Which models:** [docs/SUPPORTED_MODELS.md](docs/SUPPORTED_MODELS.md) (not “any LLM”).

| Start here | Link |
|---|---|
| What we claim | [WHAT_WE_CLAIM.md](WHAT_WE_CLAIM.md) |
| Model / method card | [MODEL_CARD.md](MODEL_CARD.md) |
| Authorship / AI-assisted workflow | [docs/AUTHORSHIP.md](docs/AUTHORSHIP.md) |
| Latest local validation | [docs/LOCAL_VALIDATION_20260729.md](docs/LOCAL_VALIDATION_20260729.md) |
| Research intent / interpretation boundary | [docs/RESEARCH_INTENT.md](docs/RESEARCH_INTENT.md) |
| What it actually gives / adjacent uses | [docs/WHAT_IT_ACTUALLY_GIVES.md](docs/WHAT_IT_ACTUALLY_GIVES.md) |
| Still unchecked | [docs/OPEN_CHECKS.md](docs/OPEN_CHECKS.md) |
| Supported families | [docs/SUPPORTED_MODELS.md](docs/SUPPORTED_MODELS.md) |
| Live vs synthetic evidence | [docs/EVIDENCE_TIERS.md](docs/EVIDENCE_TIERS.md) |
| Lab hardware limits | [docs/LAB_SCOPE.md](docs/LAB_SCOPE.md) |
| Auditor issue triage | [docs/AUDITOR_ISSUES.md](docs/AUDITOR_ISSUES.md) |
| Русская витрина | [README.ru.md](README.ru.md) |

## Quick facts (honest)

| Question | Answer |
|---|---|
| Real model tests? | **Yes** — DeepSeek-V2-Lite **and** GigaChat (10B/20B) Tier L |
| Only DeepSeek forever? | **No** — law fits DeepSeek-**family** open MoE; GigaChat verified in lab |
| Any MoE (Mixtral, …)? | **Not without a new adapter** |
| 100-task GPU HumanEval? | **Not claimed** — out of author laptop scope |
| Synthetic plots? | Tier **S** only — not live-gate proof |
| Lean code (DeepSeek v36 / GigaChat v34) | pass@1 **tie** with classic |
| Prefetch | lean HumanEval: miss-wait **ours better vs none**; short vs freq/LRU/SGD: can **FAIL** ([v1](results/misswait_baselines_v1.md), [v2](results/misswait_baselines_v2.md)) |

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

Published snapshots:

- DeepSeek: [results/v36_humaneval_prefetch_edge.md](results/v36_humaneval_prefetch_edge.md)
- GigaChat: [results/gigachat_v34_humaneval.md](results/gigachat_v34_humaneval.md), [results/gigachat_v32_lean.md](results/gigachat_v32_lean.md), [results/gigachat_v21_orbit_apply.md](results/gigachat_v21_orbit_apply.md)

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
- Authorship / how this code was produced: [docs/AUTHORSHIP.md](docs/AUTHORSHIP.md)

```bibtex
@software{moe_orbit_prefetch,
  author = {Ivanchenkov, Andrey},
  title = {moe-orbit-prefetch},
  year = {2026},
  url = {https://github.com/andreyivan4enkov/moe-orbit-prefetch},
  license = {Apache-2.0},
  version = {0.5.7}
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
