# Lab result: dynamic_weight_orbit_v13 (sanitized)

**Date:** 2026-07-24  
**Verdict:** `PASS_PHASE5A_DYNAMIC`  
**Source lab:** theory_megaattractor / `sparse-stigmergy/dynamic_weight_orbit_v13`

## What was tested

1. Local shard smoke (DeepSeek-R1-Distill-Llama-8B tensor open — environment check).
2. DeepSeek-V2-Lite-Chat `index.json`: 64 experts on layer 1.
3. Load **one** routed expert (layer=1, id=0) from safetensors shard.
4. Hot hit on second `get_expert`.
5. Sleep/evict via local mean of `S_env` masses.

## Observed (from that run)

| Check | Result |
|---|---|
| load_expert0 keys | `down_proj.weight`, `gate_proj.weight`, `up_proj.weight` |
| resident after one expert | ~17.3 MB |
| after loading more then evict | resident ~51.9 MB → ~17.3 MB; hot=1 |

## Plain reading

An expert loads as a **slice**, not the full MoE. Sleep reduces resident bytes.

## Not claimed

- End-to-end chat quality.
- Identity with DeepSeek’s trained gate.
- Reproducible wall-clock on every machine (I/O and cache dominate).
