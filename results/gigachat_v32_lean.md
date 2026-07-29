# Lab result: gigachat20_bin_orbit_v32_lean (sanitized)

**Date:** 2026-07-25  
**Verdict:** `PASS_PREFETCH_TIME_AND_WAIT`  
**Model:** `ai-sage/GigaChat-20B-A3B-instruct-v1.5-bf16`  
**Hardware:** author MacBook-class lab  
**Source lab:** `sparse-stigmergy/gigachat20_bin_orbit_v32_lean`

## Plain reading

GigaChat-20B MoE ran with lazy `.bin`/safetensors shards + orbit prefetch (spine ~3.1GB bf16 resident). Ours → classic:

- Lower expert miss-wait: **True** (wins=4, losses=0)
- Faster decode on that stand: **True** (wins=4, losses=0)
- Mean miss-wait (that run): ours ≈ 94.5 s, classic ≈ 102.6 s (CPU-bound)

## Not claimed

- Identical seconds on every machine.
- That orbit equals the trained gate.
