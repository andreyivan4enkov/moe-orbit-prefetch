# What we claim (and what we do not)

This file is the claim boundary for **Object A**: dynamic MoE expert residency
driven by an orbit predictor on embedding / residual `h`.

## We claim (verified in our labs)

1. **Complete open-source Object A tree** in this repo (MIT):
   - `OrbitPredictor`, `DynamicExpertStore`, `emergent_metrics`, `process_gate`
   - **`SparseDeepseekRuntime`** — sparse generate path with modeled expert prefetch / deposit / sleep
   - **`deepseek_chat_engine`** — chat helper on top of that runtime
   - runnable examples and the lean HumanEval/QuixBugs bench script used in lab
2. **Empirical lab results** (see `results/`):
   - Expert slice load + hit + sleep reduced resident bytes (v13).
   - Lean code bench (v36): pass@1 **tied** classic; miss-wait **ours better**; late learning↑ **False**.
3. Thresholds inside the predictor use **local field/window statistics** (not magic PASS constants).

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
