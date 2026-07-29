# dynamic_weight_orbit_v13

**verdict:** PASS_PHASE5A_DYNAMIC

## Checklist

- [PASS] r1_shard_local: shard=model-00001-of-000002.safetensors path=<HF_CACHE>/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1/model-00001-of-000002.safetensors
- [PASS] r1_dynamic_tensor: shape=(128256, 4096) bytes=1050673152
- [PASS] v2_index_local: model=deepseek-ai/DeepSeek-V2-Lite-Chat experts_layer1=64
- [PASS] v2_expert0_mapped: shard=model-00001-of-000004.safetensors n_tensors=3
- [FAIL] v2_shard1_local_complete: bytes=0 path=None
- [PASS] download_shard1: path=<HF_CACHE>/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7/model-00001-of-000004.safetensors bytes=8594887408
- [PASS] load_expert0: keys=['down_proj.weight', 'gate_proj.weight', 'up_proj.weight'] resident=17301504
- [PASS] hot_hit: hits=1
- [PASS] evict_sleep: dropped=[(1, 1), (1, 2)] resident 51904512→17301504 hot=1

## Plain reading

Сначала локально (R1 + index V2). Потом при необходимости один шард ~8.6GB.
Эксперт грузится куском, не вся модель. Sleep = evict по mean S_env.

stats={"n_hot": 1, "resident_bytes": 17301504, "bytes_loaded": 51904512, "bytes_evicted": 34603008, "n_loads": 3, "n_hits": 1, "n_misses": 3, "s_env_keys": 3}
