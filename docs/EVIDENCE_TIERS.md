# Evidence tiers (for auditors — no mixing)

This repository contains **two different kinds of evidence**. Mixing them is the usual reason Grok/Gemini say “not real tests.”

| Tier | Name | Uses live MoE weights? | What it proves | What it does **not** prove |
|---|---|---|---|---|
| **L** | Live lab | **Yes** (DeepSeek and/or GigaChat) | Residency / miss-wait / lean code on real open MoE | That offline synthetic hit equals live gate hit |
| **S** | Synthetic analysis | **No** | Field dynamics of `OrbitPredictor` on a learnable stream | Anything about the trained MoE router |

**Rule:** Tier **S** numbers must never be cited as Tier **L** proof.  
**Families:** [SUPPORTED_MODELS.md](SUPPORTED_MODELS.md).

---

## Tier L — real model (start here)

### Published lab snapshots (already run)

| Artifact | Model | Claim (honest) |
|---|---|---|
| [results/v13_dynamic_expert_slice.md](../results/v13_dynamic_expert_slice.md) | DeepSeek-V2-Lite-Chat shards | Expert slice load + hot hit + sleep reduces resident bytes |
| [results/v13_full_report.md](../results/v13_full_report.md) | same family | Longer v13 report |
| [results/v36_humaneval_prefetch_edge.md](../results/v36_humaneval_prefetch_edge.md) | DeepSeek-V2-Lite-Chat | pass@1 **tie** with classic; miss-wait **ours better** (wins>losses); late learning↑ **False** |
| [results/v36_humaneval_orbit_results.json](../results/v36_humaneval_orbit_results.json) | same | Machine-readable numbers from that run |
| [results/gigachat_v21_orbit_apply.md](../results/gigachat_v21_orbit_apply.md) | GigaChat3-10B (+ Ultra index) | Orbit store smoke on DeepSeek-style MoE |
| [results/gigachat_v32_lean.md](../results/gigachat_v32_lean.md) | GigaChat-20B | miss-wait / decode edge vs classic |
| [results/gigachat_v34_humaneval.md](../results/gigachat_v34_humaneval.md) | GigaChat-20B | lean code: pass@1 tie; miss-wait ours better |
| [results/misswait_baselines_v1.md](../results/misswait_baselines_v1.md) | DeepSeek-V2-Lite-Chat | Short miss-wait vs frequency/LRU/prev_copy/none: **FAIL** (ours worse on this stand) |

### How to re-run Tier L yourself

Weights are **not** in git (DeepSeek/HF terms). Download into Hugging Face cache first.

```bash
git clone https://github.com/andreyivan4enkov/moe-orbit-prefetch.git
cd moe-orbit-prefetch
pip install -e ".[runtime]"

# Smoke: one expert slice + sleep (minutes)
python examples/02_smoke_expert_slice.py
python examples/03_smoke_dynamic_weights_v13.py

# Chat path (needs Lite-Chat in cache)
python examples/04_chat_ask.py "Hello" --max-new 32

# Lean HumanEval/QuixBugs ours→classic (hours on CPU laptop; hardware-bound)
python examples/bench_humaneval_lean/bench_humaneval_lean.py
```

Wall-clock and absolute miss-seconds will differ by machine. The **qualitative** checks to re-verify: residency smoke works; if you run the lean bench, compare ours vs classic with the same emergent gate (`wins > losses`), not a fixed margin.

Full 50–100 task / multi-GPU evaluation is **out of author laptop scope** — [LAB_SCOPE.md](LAB_SCOPE.md).

---

## Tier S — synthetic (optional, no weights)

```bash
pip install -e ".[analysis]"
python analysis/generate_orbit_trajectory.py
python analysis/plot_orbit_analysis.py
python analysis/sensitivity_resonance_decay.py
```

See [analysis/REPORT.md](../analysis/REPORT.md). Purpose: inspect \(S_{\mathrm{env}}\) / orbit math without downloading a MoE.

---

## Known open risks (honest)

See [RISKS.md](RISKS.md). Short version:

1. History prune slack was a stand buffer (now = exact `window`).
2. Resonance decay needs sensitivity reporting (script provided; not a magic PASS).
3. Tier S is learnable by design → **weaker** signal than live gate; Tier L v36 already shows late hit did **not** rise.
