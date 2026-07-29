# Examples

| Script | Needs weights? | What it shows |
|---|---|---|
| `01_toy_orbit_no_weights.py` | No | OrbitPredictor API |
| `02_smoke_expert_slice.py` | HF cache | One expert load + hit + sleep |
| `03_smoke_dynamic_weights_v13.py` | HF cache (+ optional download) | Full v13 residency smoke |
| `04_chat_ask.py` | HF cache | End-to-end sparse chat + orbit |
| `05_gigachat_store_smoke.py` | HF cache | GigaChat 10B/20B store-path smoke |
| `bench_humaneval_lean/bench_humaneval_lean.py` | HF cache | Ours vs classic lean code bench (long) |
| `bench_misswait_baselines_v1/bench_misswait_baselines_v1.py` | HF cache | Short miss-wait: orbit → frequency/LRU/prev_copy/none |
| `bench_misswait_baselines_v2/bench_misswait_baselines_v2.py` | HF cache | + online SGD; primary + cache_sens (orbit last) |

Weights: download yourself under DeepSeek/HF terms. Not in git.
