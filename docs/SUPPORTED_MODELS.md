# Supported model families

## Short answer

| Layer | What it is | Portability |
|---|---|---|
| **Law / math** (`OrbitPredictor`, `S_env`, deposit, sleep) | Prefetch orbit from residual / embedding `h` | **Any** MoE that exposes `h` and a true expert set \(T\) after the live gate |
| **Runtime adapter** (`SparseDeepseekRuntime`, shard names, attention) | DeepSeek-style open MoE stack | **DeepSeek-family** (DeepSeek-V2/V3-style + **GigaChat** MoE builds that reuse that layout) |
| **Not claimed** | Drop-in for every MoE | Mixtral / Qwen-MoE / proprietary closed stacks need a **new adapter** |

So: the **architecture of Object A** (orbit from `h` → residency) fits similar **open** MoE networks in the DeepSeek lineage. The **shipped Python runtime** in this repo is that lineage’s weight layout — not a universal MoE engine.

## Lab evidence (author laptop)

| Model | What was measured | Artifact |
|---|---|---|
| DeepSeek-V2-Lite-Chat | Expert slice + sleep; lean HumanEval/QuixBugs miss-wait edge | `results/v13_*`, `results/v36_*` |
| GigaChat3-10B-A1.8B (bf16) | Store + orbit smoke (no full `from_pretrained`) | `results/gigachat_v21_orbit_apply.md` |
| GigaChat-20B-A3B | Lean orbit / miss-wait; lean code bench miss-wait edge, code tie | `results/gigachat_v32_lean.md`, `results/gigachat_v34_humaneval.md` |

GigaChat Ultra **702B** — index/map only on this lab (full weights do not fit). Architecture tag on Ultra index: DeepSeek-V3-style MoE.

## Why GigaChat fits

Lab notes: GigaChat MoE builds used here follow a **DeepSeek-style** causal MoE layout (routed experts + gate + safetensors/bin shards). The same Object A law applies; runtime differences are attention/RoPE details and shard formats (handled in lab adapters).

## What to tell auditors

- “Works only on DeepSeek-V2-Lite” → **too narrow** (under-claims GigaChat lab).
- “Works on any MoE / any LLM” → **too wide** (false).
- Correct: **Object A law is for open MoE with accessible experts; primary verified family = DeepSeek-style including GigaChat; packaged runtime targets that family.**

See also [LAB_SCOPE.md](LAB_SCOPE.md), [EVIDENCE_TIERS.md](EVIDENCE_TIERS.md), [MODEL_CARD.md](../MODEL_CARD.md).
