# What we claim (and what we do not)

This file is the claim boundary for **Object A**: dynamic MoE expert residency
driven by an orbit predictor on embedding / residual `h`.

**Auditors:** use [docs/EVIDENCE_TIERS.md](docs/EVIDENCE_TIERS.md) (Tier L vs Tier S),
[docs/LAB_SCOPE.md](docs/LAB_SCOPE.md) (laptop lab — no 100-task GPU claim),
and [docs/AUDITOR_ISSUES.md](docs/AUDITOR_ISSUES.md) before saying “no real tests.”

## We claim (verified in our labs — Tier L)

1. **Complete open-source Object A tree** in this repo (Apache-2.0):
   - `OrbitPredictor`, `DynamicExpertStore`, `emergent_metrics`, `process_gate`
   - **`SparseDeepseekRuntime`** — sparse generate path with modeled expert prefetch / deposit / sleep
   - **`deepseek_chat_engine`** — chat helper on top of that runtime
   - runnable examples + lean HumanEval/QuixBugs bench: `examples/bench_humaneval_lean/`
2. **Empirical lab results on real DeepSeek-V2-Lite(-Chat)** on **author laptop hardware** (see `results/`):
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
| Production inference engine / SOTA speed on every GPU | Research prototype; Mac laptop numbers are hardware-bound |
| Full HumanEval (100+) / multi-GPU sweep | **Out of author lab scope** — see LAB_SCOPE.md |
| Guaranteed rising predict-hit over long runs | v36 showed late hit did **not** beat early |
| Tier S synthetic hit = live gate hit | Offline analysis is learnable by design — see EVIDENCE_TIERS |

## Honesty rule

If a result is only structural (load/evict) or only cost (miss-wait) without answer quality, we say so. We do not publish unverified inventions. We do not relabel synthetic plots as live DeepSeek proof. We do not pretend a laptop lab is a datacenter GPU suite.
