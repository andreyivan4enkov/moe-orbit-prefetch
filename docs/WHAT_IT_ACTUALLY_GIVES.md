# What Object A actually gives (no fantasy)

This page is for people who want the **useful residue** of the work without
victory marketing.

## What you can take from this repo today (grounded)

1. **A complete sparse DeepSeek-family runtime sketch**  
   Lazy expert residency + pack cache + optional modeled prefetch + sleep/evict.  
   Useful if you study **MoE on limited RAM** (laptop / single box) and need a
   readable reference implementation — not a production server.

2. **`OrbitPredictor` as a research probe**  
   Induction + local field (`S_env`) + window resonance on residual/`h`.  
   It is a **hypothesis engine** for “does `h` carry prefetchable expert structure?”  
   It is **not** proven better than frequency/SGD on every stand (see FAIL below).

3. **Classical prefetch baselines in the same API**  
   Frequency, LRU/recent-hot, prev_copy, online SGD+sigmoid, and `none`.  
   Useful for **any** MoE-prefetch experiment that needs honest controls.

4. **Emergent comparison protocol**  
   `emerges_greater` → PASS iff `wins > losses` on paired steps.  
   Useful as a **reporting convention** that avoids magic margins.

5. **Evidence hygiene**  
   Tier L (live weights) vs Tier S (synthetic analysis), authorship note,
   hardware limits, auditor triage. Useful as a **template for honest lab notes**.

## What the numbers currently say (both sides)

| Stand | Result (short) |
|---|---|
| Lean HumanEval-style (DeepSeek v36 / GigaChat v34) | Code pass@1 **tie** vs `none`; miss-wait **ours better** vs `none` |
| Short miss-wait v1 (frequency/LRU/prev_copy/none) | Orbit **worse** on miss-wait (`FAIL`) |
| Short miss-wait v2 (SGD + orbit-last cache check) | Primary **FAIL** vs SGD/freq/none; cache_sens **MIXED** (beats freq when last, still loses to `none`) |

## Adjacent uses that are **reasonable to try** (not promises)

These are **narrow, technical** reuse ideas. They are **not** claims that the
method already works there.

| Idea | Why it is plausible | Why it is **not** proven here |
|---|---|---|
| Reuse `DynamicExpertStore` for other DeepSeek-style open MoE on CPU/RAM-limited hosts | Same safetensors expert slicing problem | Other families need adapters; not tested |
| Use classic predictors as controls in someone else’s prefetch paper | Same API, small code | Their hardware/results will differ |
| Use `emerges_greater` in other A/B cost or hit series | Threshold-free majority vote | Not a universal science method claim |
| Study whether residual `h` correlates with next experts (diagnostics) | Orbit/SGD both read `h` | Correlation ≠ deployable win |

## Adjacent ideas we **refuse** to advertise

- “This proves a new theory of intelligence.”
- “Drop-in faster than all MoE systems.”
- “Works on any model / any GPU cluster.”
- “Short FAIL means the lean PASS was fake” **or** the reverse — different stands.

## Reading rule

If a sentence cannot be tied to a file under `results/` or a runnable example,
treat it as **interpretation**, not deliverable value.
