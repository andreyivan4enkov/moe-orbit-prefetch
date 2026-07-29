# Lab result: gigachat_orbit_apply_v21 (sanitized)

**Date:** 2026-07-24  
**Verdict:** `PASS_GIGACHAT_ORBIT`  
**Hardware:** author MacBook-class lab — [docs/LAB_SCOPE.md](../docs/LAB_SCOPE.md)  
**Source lab:** `sparse-stigmergy/gigachat_orbit_apply_v21`

## Plain reading

- GigaChat Ultra **702B**: only **index/config** (full weights do not fit the laptop). Index shows DeepSeek-V3-style MoE (256 routed experts, top-8).
- Applied live store smoke on **GigaChat3-10B-A1.8B-bf16**: expert load by orbit, FFN non-zero, hot hit on repeat `get_expert`. Full MoE not loaded via `from_pretrained`.

## Not claimed

- Chat quality SOTA on 10B.
- Running 702B generate on this machine.
