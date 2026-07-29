# NOTICE — third party and models

## This repository (MIT)

All Python modules, examples, docs, and lab write-ups in this git tree are released under the MIT License (see [LICENSE](LICENSE)).

## Model weights (not shipped)

Examples that call DeepSeek download or read weights from the Hugging Face Hub, for example:

- `deepseek-ai/DeepSeek-V2-Lite-Chat`
- optionally `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (environment smoke in example 03)

Those weights remain under **DeepSeek / Hugging Face model licenses and terms**.  
This project does **not** grant rights to the weights. Download and use them yourself under their terms.

## Architecture notes

`sparse_moe_runtime.py` implements a **sparse** DeepSeek-V2-Lite-style forward (MLA/YaRN-style RoPE, MoE gate, expert SwiGLU) for research residency experiments. It is our runtime assembly for Object A, not an official DeepSeek product binary. Behavior should be validated against your deployment; we publish lab measurements, not a warranty of bit-identical logits vs `transformers` `from_pretrained`.

## Prior art

See [RELATED_WORK.md](RELATED_WORK.md). MoE, expert offload, and prefetch as topics predate this repository.
