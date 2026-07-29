# Model Card — moe-orbit-prefetch (Object A)

Following the spirit of [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
and common open-source ML repo practice. This is a **method + runtime card**, not a foundation model card.
Weights are **not** shipped; use DeepSeek/HF weights under their terms.

## Model / method details

| Field | Value |
|---|---|
| Name | `moe-orbit-prefetch` (Object A) |
| Version | 0.5.2 |
| Type | Research prototype: MoE expert **prefetch / residency** from residual `h` |
| License | Apache-2.0 (+ [ATTRIBUTION.md](ATTRIBUTION.md) for substantial products) |
| Primary target | DeepSeek-V2-Lite / Lite-Chat **and** GigaChat MoE in the same open DeepSeek-style family |
| Lab also measured | GigaChat3-10B smoke; GigaChat-20B lean miss-wait + lean code bench — see `results/gigachat_*` |
| Not included | Model weights, multi-GPU serving, quantized experts, Object B (chat topology), arbitrary Mixtral/Qwen-MoE without a new adapter |

## Intended use

- Study / fork dynamic expert load from an orbit predictor on `h`.
- Reproduce lean lab smokes and published lean code benches on a laptop-class machine.
- Apply Object A to **open MoE** networks that expose experts (DeepSeek-style / GigaChat).
- Compare miss-wait / residency vs a classic baseline (**ours first**, then classic).

## Out of scope

- Replacing the model’s trained MoE gate / router.
- Claiming SOTA HumanEval or production inference latency.
- Claiming **every** MoE family without an adapter — see [docs/SUPPORTED_MODELS.md](docs/SUPPORTED_MODELS.md).
- Treating Tier **S** (synthetic analysis) as Tier **L** (live model) proof — see [docs/EVIDENCE_TIERS.md](docs/EVIDENCE_TIERS.md).

## Lab hardware (author)

| Item | Value |
|---|---|
| Author lab | Apple MacBook-class laptop (CPU / unified memory, ~16GB class) |
| Full 50–100 task GPU suite | **Out of scope** for this lab (resource-bound) |
| Published Tier L stand | Lean HumanEval/QuixBugs subset on **DeepSeek-V2-Lite** and **GigaChat-20B** |
| Absolute wall-clock | Hardware-bound; re-check qualitative gates, not raw seconds |

## Evaluation data

| Tier | Artifact | Notes |
|---|---|---|
| L | `results/v13_*`, `results/v36_*` | Real DeepSeek-V2-Lite(-Chat) |
| L | `results/gigachat_v21_*`, `v32_*`, `v34_*` | Real GigaChat 10B/20B lab |
| S | `analysis/` | Synthetic learnable stream for field plots only |

Honest lean summary (both DeepSeek v36 and GigaChat v34): pass@1 **tie** with classic; miss-wait **ours better** (wins>losses).

## Authorship / development process

This repository uses an **AI-assisted implementation workflow**. The maintainer's primary role is architectural and analytical: problem framing, risk/claim boundaries, technical-task writing, acceptance criteria, and review. Substantial code and documentation were produced through LLM coding agents under that direction.

See [docs/AUTHORSHIP.md](docs/AUTHORSHIP.md).

## Ethical considerations

- Do not redistribute DeepSeek weights via this repo.
- Do not overclaim safety or capability gains beyond measured miss-wait / residency.
- Prefetch bugs can increase I/O; see [docs/AUDITOR_ISSUES.md](docs/AUDITOR_ISSUES.md).

## Citation

See [CITATION.cff](CITATION.cff).
