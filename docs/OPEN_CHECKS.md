# What is still unchecked (MacBook lab honesty)

Last updated: 2026-07-29 (after v0.5.6 / into v0.5.7).

This list is for reviewers. **Unchecked ≠ secretly proven.**  
If it is not here as DONE, do not assume it passed.

## Done on this laptop (Tier L / tooling)

| Item | Evidence |
|---|---|
| Expert slice load + hit + sleep | `results/v13_*` |
| Lean HumanEval/QuixBugs: orbit vs **none** (no modeled prefetch) | `results/v36_*`, GigaChat v32/v34 |
| Short miss-wait vs frequency / LRU / prev_copy / none | `results/misswait_baselines_v1.md` → **FAIL** for orbit |
| Self-deadlock in `relieve`+`store._lock` | fixed in code + v0.5.6 notes |
| Unit tests (store coalesce, metrics, classic API) | `tests/` |
| GigaChat store smoke (10B; 20B may SKIP if shards absent) | `examples/05_*` |

## Newly addressed in v0.5.7 (this cut)

| Item | Status |
|---|---|
| Online **SGD + sigmoid** classical predictor in live miss-wait path | **DONE** — primary: orbit did **not** beat SGD under `wins>losses` (1–1) |
| Order / OS-cache sensitivity (orbit last after warm classics) | **DONE** — MIXED: beat frequency when last; still lost to `none` |
| Public “what it actually gives” + grounded adjacent uses | `docs/WHAT_IT_ACTUALLY_GIVES.md` |
| Tier S resonance sensitivity table regenerated | `analysis/figures/sensitivity_resonance.json` |

## Still OPEN (not claimed)

| Item | Why still open |
|---|---|
| Full HumanEval (100+) / multi-GPU | Out of author lab scope (`LAB_SCOPE.md`) |
| Head-to-head vs PowerInfer / ExpertFlow / Fate on equal GPU | No equal hardware here |
| Mixtral / Qwen-MoE / other MoE adapters | Needs new pack/attention adapter |
| Cold OS page-cache flush (true disk-cold every mode) | macOS cannot reliably purge all without reboot; order-swap is only a **partial** control |
| Long-horizon learning curve on live gate (hours+) | v36 already: late learning↑ **False**; longer study OPEN |
| Predictor lock / multi-writer formal proof | Documented risk; not fully redesigned |
| Quantized expert path | Not implemented |
| Answer-quality A/B on short miss-wait stand | Short stand measured **miss-wait**, not pass@1 |
| Re-run v1 with fully interleaved modes on identical cache | Time/thermal limited; OPEN for external labs |

## How to read mixed results

- Lean long code stand: orbit miss-wait **better than none**, code **tie**.
- Short v1 stand: orbit miss-wait **worse than frequency/LRU/prev_copy/none**.
- These do **not** cancel each other by magic. They are **different questions**.
  Public docs must show **both**.
