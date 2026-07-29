# Model Card — moe-orbit-prefetch (Object A)

Following the spirit of [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
and common open-source ML repo practice. This is a **method + runtime card**, not a foundation model card.
Weights are **not** shipped; use DeepSeek/HF weights under their terms.

## Model / method details

| Field | Value |
|---|---|
| Name | `moe-orbit-prefetch` (Object A) |
| Version | 0.5.0 |
| Type | Research prototype: MoE expert **prefetch / residency** from residual `h` |
| License | Apache-2.0 (+ [ATTRIBUTION.md](ATTRIBUTION.md) for substantial products) |
| Primary target | DeepSeek-V2-Lite / Lite-Chat (sparse safetensors path) |
| Not included | Model weights, multi-GPU serving, quantized experts, Object B (chat topology) |

## Intended use

- Study / fork dynamic expert load from an orbit predictor on `h`.
- Reproduce lean lab smokes and the published lean code bench on a laptop-class machine.
- Compare miss-wait / residency vs a classic baseline (**ours first**, then classic).

## Out of scope

- Replacing DeepSeek’s trained MoE gate / router.
- Claiming SOTA HumanEval or production inference latency.
- Treating Tier **S** (synthetic analysis) as Tier **L** (live model) proof — see [docs/EVIDENCE_TIERS.md](docs/EVIDENCE_TIERS.md).

## Lab hardware (author)

| Item | Value |
|---|---|
| Author lab | Apple MacBook-class laptop (CPU / unified memory, ~16GB class) |
| Full 50–100 task GPU suite | **Out of scope** for this lab (resource-bound) |
| Published Tier L stand | Lean HumanEval/QuixBugs subset (see `results/v36_*`) |
| Absolute wall-clock | Hardware-bound; re-check qualitative gates, not raw seconds |

This matches common practice: publish what you can reproduce on the author’s hardware and label larger suites as future / external compute.

## Evaluation data

| Tier | Artifact | Notes |
|---|---|---|
| L | `results/v13_*`, `results/v36_*` | Real DeepSeek-V2-Lite(-Chat) |
| S | `analysis/` | Synthetic learnable stream for field plots only |

Honest v36 summary: pass@1 **tie** with classic; miss-wait **ours better** (wins>losses); late learning↑ **False**.

## Ethical considerations

- Do not redistribute DeepSeek weights via this repo.
- Do not overclaim safety or capability gains beyond measured miss-wait / residency.
- Prefetch bugs can increase I/O; see [docs/AUDITOR_ISSUES.md](docs/AUDITOR_ISSUES.md).

## Citation

See [CITATION.cff](CITATION.cff).
