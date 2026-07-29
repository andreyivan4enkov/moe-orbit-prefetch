# bench_misswait_baselines_v1

stamp: `20260729_105315`
hardware: MacBook Pro 2019 Intel i9 / 16 GB (throttling)
max_new=8 n_prompts=2
stop_reason: `none`

## Meaning

Same short live prompts on DeepSeek-V2-Lite-Chat. Measure how long we wait
for missing experts (miss-wait). Ours = OrbitPredictor. Classics = frequency,
LRU/recent-hot, prev_copy, and none (no modeled prefetch).

## Per-mode mean miss-wait

| mode | family | mean miss-s | n ok |
|---|---|---:|---:|
| orbit | ours | 179.36 | 2 |
| frequency | classic | 115.39 | 2 |
| lru | classic | 157.07 | 2 |
| prev_copy | classic | 156.63 | 2 |
| none | classic | 148.95 | 2 |

## Ours vs classic (lower miss better; PASS iff wins > losses)

- `frequency`: **FAIL** wins=0 losses=2 mean_ours=179.36s mean_frequency=115.39s
- `lru`: **FAIL** wins=0 losses=2 mean_ours=179.36s mean_lru=157.07s
- `prev_copy`: **FAIL** wins=0 losses=2 mean_ours=179.36s mean_prev_copy=156.63s
- `none`: **FAIL** wins=0 losses=2 mean_ours=179.36s mean_none=148.95s

## Verdict: `FAIL_OURS_NOT_BETTER_ON_COMPARED`

This is a short laptop stand, not a multi-GPU HumanEval marathon.
Qualitative claim only: whether orbit miss-wait beats these classical
predictors on the same live prompts under emergent wins>losses.

JSON: `bench_misswait_baselines_v1_20260729_105315_results.json`
Log: `bench_misswait_baselines_v1_20260729_105315.log`

