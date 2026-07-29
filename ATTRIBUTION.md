# Attribution (how to use this work)

**License:** [Apache License 2.0](LICENSE) — free to use, modify, fork, and ship. **No payment required.**

This file explains the **attribution expectation** the author asks for. Apache-2.0 already requires keeping copyright / NOTICE on redistribution. Below is how we interpret “small task” vs “inside a larger system”.

## Small / personal / research snippets

If you:

- run the examples,
- copy a function into a notebook,
- experiment on your laptop,
- cite us in a paper optionally,

then **Apache-2.0 alone is enough**: keep `LICENSE` + `NOTICE` when you redistribute source, and edit freely.

## Substantial systems / products / forks used as a method

If you incorporate this **orbit-prefetch / dynamic expert residency method** into a larger system (product, cloud inference stack, another open-source framework’s core path, a public fork marketed as an MoE runtime), please:

1. Keep `LICENSE` and this repository’s `NOTICE` text in your distribution (Apache §4).
2. State clearly that the method / module comes from:
   - **Repository:** https://github.com/andreyivan4enkov/moe-orbit-prefetch  
   - **Method name:** Object A — MoE orbit prefetch from embedding/residual `h`  
   - **Author:** andreyivan4enkov
3. Prefer a short credit line in docs or UI “About”, for example:

> Expert residency / orbit prefetch based on  
> [moe-orbit-prefetch](https://github.com/andreyivan4enkov/moe-orbit-prefetch)  
> (Apache-2.0; andreyivan4enkov).

4. Academic citation: use [CITATION.cff](CITATION.cff).

You may still charge for **your** product or support. The author does **not** charge a license fee for this code. Attribution is about **origin of the method**, not about money.

## Forks

Forks on GitHub already show lineage. If you publish a fork as a standalone product, keep NOTICE and mention this upstream (or “based on moe-orbit-prefetch”).

## What you must not do

- Claim you invented MoE / DeepSeek’s gate / expert offload as a field (see [RELATED_WORK.md](RELATED_WORK.md)).
- Strip NOTICE while shipping a substantial derivative.
- Redistribute **model weights** under this license (weights stay under DeepSeek/HF terms).

## Contact

Issues on the GitHub repository.
