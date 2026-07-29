# Short miss-wait baselines v2 (SGD + cache-order check)

**Honest mixed evidence — not a marketing win.**

## Meaning

We still measured **miss-wait** (how long we wait for missing experts) on the same
two short DeepSeek-V2-Lite-Chat prompts (`max_new=8`).

Two phases:

1. **primary** (ours first): orbit → online SGD → frequency → none  
2. **cache_sens** (orbit last): frequency → none → orbit  
   Partial OS shard-cache check only — **not** a true disk-cold reboot control.

## Primary (ours first)

Source log: `bench_misswait_baselines_v2_20260729_122829.log`  
(process died mid `cache_sens`; primary finished cleanly).

| mode | mean miss-s | per-prompt |
|---|---:|---|
| orbit | 138.40 | 124.60, 152.20 |
| sgd | 142.38 | 145.11, 139.65 |
| frequency | 116.46 | 138.15, 94.76 |
| none | 120.65 | 113.57, 127.72 |

Ours vs classic (`wins > losses` on lower miss):

| classic | wins | losses | result |
|---|---:|---:|---|
| sgd | 1 | 1 | **FAIL** (tie; not wins>losses) |
| frequency | 1 | 1 | **FAIL** |
| none | 0 | 2 | **FAIL** |

**Primary verdict: `FAIL_OURS_NOT_BETTER_ON_COMPARED`**

Note: orbit mean is slightly better than SGD mean (138 vs 142), but the
emergent gate still fails because it is not a majority of per-prompt wins.

## cache_sens (orbit last)

Source: `reports/bench_misswait_baselines_v2_20260729_135421_*`

| mode | mean miss-s | per-prompt |
|---|---:|---|
| frequency | 140.52 | 137.68, 143.36 |
| none | 104.90 | 101.70, 108.09 |
| orbit | 130.79 | 134.29, 127.29 |

| classic | wins | losses | result |
|---|---:|---:|---|
| frequency | 2 | 0 | **PASS** (orbit better after warm cache) |
| none | 0 | 2 | **FAIL** (none still better) |

**cache_sens verdict: `MIXED`**

## What this does / does not mean

- Does **not** erase lean HumanEval miss-wait edge vs `none`.
- Does **not** prove orbit beats strong classics on short chat.
- **Does** show OS-order matters: when orbit ran last, it beat frequency
  (unlike the colder primary where frequency won overall).
- Even with warm cache, orbit still **lost to `none`** on this short stand —
  so “prefetch always helps” is **false** here.
- Online SGD is now a live baseline; orbit did not clearly beat it under
  `wins > losses`.

## Artifacts

- Script: `examples/bench_misswait_baselines_v2/`
- Combined machine JSON for cache_sens run: `results/misswait_baselines_v2_cache_sens_results.json`
- Primary numbers reconstructed from the completed primary section of the first log
  (see also `results/misswait_baselines_v2_primary_from_log.json`)

## Ontology checklist

- [x] Ours = OrbitPredictor on `h`
- [x] SGD / frequency / none labeled classic
- [x] Primary phase: ours first
- [x] No oracle deposit
- [x] Emergent `wins > losses`
- [x] Failures published without spin
