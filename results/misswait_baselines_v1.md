# Short live miss-wait vs classical predictors (v1)

**Verdict: `FAIL_OURS_NOT_BETTER_ON_COMPARED`**

On this MacBook stand, OrbitPredictor did **not** beat classical prefetch
predictors on miss-wait (`wins > losses` failed for every compared classic).

## Meaning (plain)

We asked the same short live questions on DeepSeek-V2-Lite-Chat and measured
how long the runtime waited for missing experts to load from disk.

- **Ours** = orbit field from residual/`h`
- **Classics** = frequency, LRU/recent-hot, prev_copy, and none (no modeled prefetch)

On this short run, the classics waited **less**. That is a real local loss, not a spin.

## Why it still matters

1. Closes part of the “weak baselines” auditor gap: frequency / LRU / prev_copy
   are now in the live path, not only synthetic analysis.
2. Shows the pipeline can run ours → classic without hanging (deadlock in
   `relieve`+`store._lock` was found and fixed during this stand).
3. Does **not** erase earlier lean HumanEval miss-wait edges; those were a
   different stand (longer code tasks, classic = no modeled prefetch only).

## Setup

| Item | Value |
|---|---|
| Machine | MacBook Pro 2019, Intel i9, 16 GB, Radeon 4 GB (thermal throttling) |
| Model | DeepSeek-V2-Lite-Chat (lean fp16 spine) |
| Prompts | 2 short coding prompts |
| max_new | 8 |
| Order | orbit → frequency → lru → prev_copy → none |
| Gate | lower miss-wait better; PASS iff wins > losses |

## Mean miss-wait

| mode | family | mean miss-s | n ok |
|---|---|---:|---:|
| orbit | ours | 179.36 | 2 |
| frequency | classic | 115.39 | 2 |
| lru | classic | 157.07 | 2 |
| prev_copy | classic | 156.63 | 2 |
| none | classic | 148.95 | 2 |

## Pairwise (ours vs each classic)

| classic | wins | losses | ours better? |
|---|---:|---:|---|
| frequency | 0 | 2 | no |
| lru | 0 | 2 | no |
| prev_copy | 0 | 2 | no |
| none | 0 | 2 | no |

## Caveats (honesty)

1. **OS shard cache confound:** orbit ran first on a colder file cache; later
   modes may benefit from warm safetensors pages. This can inflate the gap
   against ours. A cold-cache re-run / interleaved order is future work.
2. **Tiny N:** 2 prompts × 8 tokens — diagnostic, not a 100-task claim.
3. Orbit had **higher** `modeled_hit` counts but still worse miss-seconds —
   wrong/extra prefetch can compete for I/O under `pack_dev_cap`.

## Artifacts

- Script: `examples/bench_misswait_baselines_v1/bench_misswait_baselines_v1.py`
- Report: `examples/bench_misswait_baselines_v1/reports/bench_misswait_baselines_v1_20260729_105315_report.md`
- JSON: `examples/bench_misswait_baselines_v1/reports/bench_misswait_baselines_v1_20260729_105315_results.json`

## Ontology checklist

- [x] Ours predicts from embedding topology (`OrbitPredictor`)
- [x] Classics are labeled classic, not sold as ours
- [x] No oracle deposit
- [x] Ours first, then classic; ours completed before classic arms
- [x] Emergent gate `wins > losses` (no magic margin)
- [x] Honest FAIL recorded
