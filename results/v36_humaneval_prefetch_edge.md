# Lab result: deepseek_humaneval_orbit_v36_lean (sanitized)

**Date:** 2026-07-26  
**Verdict:** `PASS_CODE_TIE_PREFETCH_EDGE`  
**Model:** DeepSeek-V2-Lite-Chat, sparse runtime + OrbitPredictor prefetch  
**Source lab:** theory_megaattractor / `sparse-stigmergy/deepseek_humaneval_orbit_v36_lean`

## Summary (plain)

- **Code quality:** ours pass@1 = classic pass@1 = **0.50** (3/6 tasks).
- **Expert wait:** ours lower miss-wait than classic (**wins=5, losses=1**).
- Mean miss wait (that run): ours ≈ 425.8 s, classic ≈ 461.7 s (CPU / laptop bound).
- **Orbit learning improves late vs early:** **False** (wins=1, losses=5; early hit≈0.452, late≈0.392).

## Per-task (ours vs classic pass, miss seconds)

| task | ours | classic | ours miss-s | classic miss-s |
|---|---:|---:|---:|---:|
| QuixBugs/bitcount | 1 | 1 | 245.9 | 274.6 |
| QuixBugs/gcd | 1 | 1 | 230.8 | 248.9 |
| QuixBugs/is_valid_parenthesization | 0 | 0 | 630.9 | 629.6 |
| HumanEval/0 | 0 | 0 | 567.9 | 636.9 |
| HumanEval/1 | 0 | 0 | 696.7 | 784.3 |
| HumanEval/2 | 1 | 1 | 182.4 | 195.8 |

## Plain reading

Prefetch helped **waiting for experts**, not magic code accuracy. Learning curve of the predictor did **not** improve toward the end in this lean run.

## Not in this public repo

Full generate runtime / HumanEval harness (large, machine-specific). Results are published as **evidence of what was measured**, not as a one-click reproduce of the multi-hour bench.
