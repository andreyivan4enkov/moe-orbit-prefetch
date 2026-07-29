# bench_misswait_baselines_v2

stamp: `20260729_135421`
hardware: MacBook Pro 2019 Intel i9 / 16 GB (throttling)
max_new=8 n_prompts=2 phase=cache_sens

## Meaning

Primary: ours first, then online SGD, frequency, none.
cache_sens: classics first, orbit last (partial OS-cache check only).

### Phase `cache_sens`

order: frequency → none → orbit
stop_reason: `none`

| mode | family | mean miss-s | n ok |
|---|---|---:|---:|
| frequency | classic | 140.52 | 2 |
| none | classic | 104.90 | 2 |
| orbit | ours | 130.79 | 2 |

Ours vs classic (lower miss better):

- `frequency`: **PASS** wins=2 losses=0 mean_ours=130.79s mean_frequency=140.52s
- `none`: **FAIL** wins=0 losses=2 mean_ours=130.79s mean_none=104.90s

Phase verdict: `MIXED`

## Overall: `cache_sens:MIXED`

Not a multi-GPU HumanEval. Honest laptop diagnostic only.
JSON: `bench_misswait_baselines_v2_20260729_135421_results.json`
Log: `bench_misswait_baselines_v2_20260729_135421.log`

