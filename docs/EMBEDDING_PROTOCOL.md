# Protocol: predictor ↔ experts via embeddings (MoE only)

Copied/adapted from the lab `EMBEDDING_PROTOCOL.md` for Object A.

## Policy

`architecture_policy: moe_only`. This repository does not target dense-only stands as the primary path.

## Native DeepSeek MoE

Router input = **hidden state** of the token (stream embedding after attention).  
Output = scores over experts → top-k → only selected experts run.

There is no separate “tiny LLM” in the original: the router is a layer in the same forward.

## Our extension (“spill / prefetch”)

Same channel — **embeddings** — but:

1. Predictor looks at \(h_t\) (and optional history).
2. Predicts experts on a horizon ahead.
3. Enqueues prefetch before the layer asks for weights.
4. MoE exec uses already-resident weights; miss = extra I/O.

**Dynamic load (v13):** do not load the full MoE into RAM.  
Index → spine (shared/router/attn as needed) → routed experts only from orbit \(O(h)\).  
Cold eviction = sleep by \(S_{\mathrm{env}}\). See [DESIGN_DYNAMIC_WEIGHTS.md](DESIGN_DYNAMIC_WEIGHTS.md).

Message interface: vector \(h \in \mathbb{R}^{d}\) → expert ids / distribution.  
Not vocabulary logits, not text.

## Math

Full equations: [MATH.md](MATH.md).

## Stands (lab context)

| Role | Model |
|---|---|
| Long-term target | DeepSeek-class giant MoE |
| Local MoE | DeepSeek-V2-Lite / Lite-Chat |
| Mac 16GB | sparse runtime + orbit (full weights never all resident) |
