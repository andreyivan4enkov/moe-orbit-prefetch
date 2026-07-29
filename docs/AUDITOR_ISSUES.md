# Auditor issues (Grok / Gemini) — triage vs this repo

Last updated: 2026-07-29 (v0.5 workstream).  
Companion: [EVIDENCE_TIERS.md](EVIDENCE_TIERS.md), [RISKS.md](RISKS.md).

Legend: **FIXED** in this tree · **PARTIAL** · **OPEN** (accepted) · **OVERSTATED** (code already contradicts claim) · **BY_DESIGN** (Object A = DeepSeek Lite path, not universal MoE)

---

## 1. Critical correctness

| ID | Claim | Verdict | Action |
|---|---|---|---|
| 1.1 | Double-load race in `get_expert` | **PARTIAL** — insert path already double-checks under lock; duplicate **I/O** still possible | **FIXED** coalesce via per-key loading Event |
| 1.2 | `s_env==0` hot forever | **OVERSTATED** — unused experts are not in `hot`; `mass < thr` already drops zeros | **FIXED** explicit cold-first + `drop_expert` API |
| 1.3 | Wrong residual in `encode_moe_h` | **OVERSTATED** for current code — intermediate MoE layers **are** executed before target | Keep regression test; mark closed unless repro |
| 1.4 | Silent prefetch `except: pass` | **FIXED** — counters + last error + optional fail-fast | done in v0.5 |
| 1.5 | DeepSeek hardcoding | **BY_DESIGN** for *packaged* runtime; **family** includes GigaChat (lab) — not all MoE | see SUPPORTED_MODELS; more adapters OPEN |

## 2. Concurrency / reliability

| ID | Claim | Verdict | Action |
|---|---|---|---|
| 2.1 | Deadlock risk one lock | **PARTIAL** — self-deadlock in `relieve`+`store._lock` around `evict_below_mean` **FIXED** (v0.5.6); other lock pairs still OPEN | v0.5.6 + v0.6 split locks |
| 2.2 | Cancel does not stop worker I/O | **FIXED** — cancel bumps epoch, drains queue, pauses | v0.5 |
| 2.3 | Predictor shared-state race | **OPEN** | v0.6 document single-writer or lock |
| 2.4 | Drain / task_done inconsistency | **PARTIAL** — drain exists; hardened with cancel | v0.5+ |

## 3. Memory / OOM

| ID | Claim | Verdict | Action |
|---|---|---|---|
| 3.1 | CUDA load without VRAM guard | **OPEN** (lab is mostly CPU) | v0.5: refuse cuda device unless guard env set; document |
| 3.2 | `resident_bytes` via `id(t)` | **OPEN** (approx) | document; improve later |
| 3.3 | No forced drop | **FIXED** — `drop_expert` / `drop_all_hot` | v0.5 |
| 3.4 | pack_dev growth | **PARTIAL** — LRU cap exists | monitor |

## 4. Math / sensitivity

| ID | Claim | Verdict | Action |
|---|---|---|---|
| 4.1 | Synthetic ≠ live learning | **ACCEPTED** — Tier L v36 late learning↑ False | EVIDENCE_TIERS; research OPEN |
| 4.2 | Heuristics / no sensitivity | **PARTIAL** — `+8` removed; sensitivity script added | more ablations OPEN |
| 4.3 | Cold-start hash on `h[0]` | **OPEN** | diversify cold-start later |
| 4.4 | NaN in metrics | **FIXED** — sanitize in `emerges_greater` | v0.5 |

## 5. Validation / generality

| ID | Claim | Verdict | Action |
|---|---|---|---|
| 5.1 | Tiny HumanEval lean bench | **ACCEPTED** — honest lean laptop snapshot; 100-task GPU **out of LAB_SCOPE** | external compute welcome |
| 5.1b | Weak classical baselines | **PARTIAL** — live frequency/LRU/prev_copy miss-wait stand added; orbit **lost** on short MacBook run | results/misswait_baselines_v1.md; cold-cache re-run OPEN |
| 5.2 | Kind synthetic GT | **ACCEPTED** | Tier S labeled |
| 5.3 | No long-horizon study | **OPEN** | later |

## 6. Tests / observability

| ID | Claim | Verdict | Action |
|---|---|---|---|
| 6.1 | Minimal tests | **PARTIAL** — add store concurrent + metrics | more OPEN |
| 6.2 | Stats not atomic | **OPEN** | later |
| 6.3 | Weak telemetry | **OPEN** | later |

## 7. Performance / scale

| ID | Claim | Verdict | Action |
|---|---|---|---|
| 7.1–7.3 | Queue/worker/multi-GPU | **OPEN** / **BY_DESIGN** for prototype | roadmap |
| 7.4 | `process_gate` fragility | **OPEN** | document limits |

## 8. Misc

| ID | Claim | Verdict |
|---|---|---|
| MIT leftovers | **FIXED** RELATED_WORK | |
| RU/EN comments | OPEN cosmetic | |
| No CI | OPEN | |
| Quantized experts | OPEN | |

---

## Roadmap (official)

**v0.5 (this cut):** 1.1 coalesce, 1.2/3.3 drop+cold-first, 1.4 errors, 2.2 cancel, 3.1 cuda guard, 4.4 NaN, tests, this doc.

**v0.6–0.7:** prefetch redesign, larger benches, model abstraction, residual audit tests, predictor locking.

**Later:** multi-device, quantized experts, CI, long-context eval.

We do **not** close 4.1/5.1 by inventing better live hit numbers. Those stay honest failures/limits until new Tier L evidence exists.
