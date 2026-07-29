# Lab result: gigachat20_humaneval_audit_v34 (sanitized)

**Date:** 2026-07-26  
**Verdict:** `PASS_CODE_TIE_PREFETCH_EDGE`  
**Model:** GigaChat-20B-A3B (lean sparse/orbit path)  
**Hardware:** author MacBook-class lab  
**Source lab:** `sparse-stigmergy/gigachat20_humaneval_audit_v34`

## Plain reading

Same lean industrial stand as DeepSeek v36 (3 HumanEval + 3 QuixBugs), ours → classic:

| Metric | Result |
|---|---|
| pass@1 | ours **0.67** (4/6) = classic **0.67** (tie) |
| miss-wait | ours better (**wins=6, losses=0**) |
| Mean miss-wait (that run) | ours ≈ 736 s, classic ≈ 794 s |

## Per-task

| task | ours | classic | ours miss-s | classic miss-s |
|---|---:|---:|---:|---:|
| QuixBugs/bitcount | 1 | 1 | 309.1 | 334.2 |
| QuixBugs/gcd | 1 | 1 | 258.5 | 284.7 |
| QuixBugs/is_valid_parenthesization | 1 | 1 | 449.0 | 486.6 |
| HumanEval/0 | 0 | 0 | 1403.7 | 1492.1 |
| HumanEval/1 | 0 | 0 | 1397.1 | 1544.0 |
| HumanEval/2 | 1 | 1 | 600.0 | 619.8 |

## Not claimed

- Full HumanEval-100 / GPU sweep.
- Rising late gate-hit (not the claim of this stand).
