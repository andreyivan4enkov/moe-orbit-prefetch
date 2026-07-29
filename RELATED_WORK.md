# Related work (prior art we reuse / cite)

This project builds **on** the following lines. Citing them is mandatory for honest reading.

## Mixture-of-Experts

- Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (2017).
- Fedus et al., *Switch Transformers* (2021).
- DeepSeek-AI, DeepSeek-V2 technical report / model cards on Hugging Face (`deepseek-ai/DeepSeek-V2-Lite`, `DeepSeek-V2-Lite-Chat`).

## Expert offload / prefetch / dynamic residency (examples)

- PowerInfer (hot/cold neurons; SOSP 2024 / arXiv:2312.12456).
- DynamicInfer and related dynamic MoE/neuron residency work.
- Broader literature on MoE expert parallelism, offload, and speculative / predicted expert loading (ExpertFlow, FATE, Pre-Attention, Speculating Experts, PROBE, … — names evolve; treat as field, not endorsement of any single paper’s identity with our code).

## What is different here

We publish a **concrete open implementation** of:

`h → OrbitPredictor (S_env + local stats) → prefetch candidates → safetensors expert slice load → deposit from true experts → sleep/evict`

and lab reports for that assembly. We do **not** claim this is the first paper on MoE prefetch.

## Models / licenses

Weights stay on Hugging Face under DeepSeek’s terms. This repo ships **code only** (Apache-2.0).
