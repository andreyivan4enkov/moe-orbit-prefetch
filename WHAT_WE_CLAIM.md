# What we claim (and what we do not)

This file is the claim boundary for **Object A**: dynamic MoE expert residency
driven by an orbit predictor on embedding / residual `h`.

## We claim (verified in our labs)

1. **Software assembly** in this repo:
   - `OrbitPredictor`: online field `S_env` + induction from `(h, tok_id)` history → top-k expert **candidates for prefetch**;
   - `DynamicExpertStore`: load **one expert’s tensors** from safetensors shards via `index.json`, hit cache, **evict** by local mean of `S_env` (sleep).
2. **Empirical lab results** (see `results/`):
   - Expert slice load + hit + sleep reduced resident bytes in the v13 smoke (`PASS_PHASE5A_DYNAMIC`).
   - On a DeepSeek-V2-Lite HumanEval/QuixBugs lean bench (v36): **code pass@1 tied** classic (0.50); **expert miss-wait lower** for ours (wins=5, losses=1). Orbit “learning improves to the end” was **False** in that run.
3. Thresholds inside the predictor use **local field/window statistics** (not magic PASS constants). Structural facts (e.g. 64 routed / top-6 on V2-Lite) are allowed.

## We do **not** claim

| Topic | Why not |
|---|---|
| Mixture-of-Experts | Prior art (Shazeer, Switch, DeepSeek, …) |
| Live DeepSeek gate / router identity | Our predictor **does not replace** the trained gate; deposit uses **true** experts from the live gate when wired |
| Inventing expert offload / prefetch | Active research field — see RELATED_WORK.md |
| Chat topological memory / RLM / Object B | **Out of scope** for this repository |
| Production inference engine / SOTA speed on every GPU | Research prototype; Mac 16GB CPU numbers are hardware-bound |
| Guaranteed rising predict-hit over long runs | v36 showed late hit did **not** beat early |

## Honesty rule

If a result is only structural (load/evict) or only cost (miss-wait) without answer quality, we say so. We do not publish unverified inventions.
