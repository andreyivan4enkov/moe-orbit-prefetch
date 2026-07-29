# Known risks (Object A) — honest status

Auditors (Grok / Gemini / humans) often repeat three points. Status below is **official** for this repo.

## R1 — History prune `window + 8`

| | |
|---|---|
| **Complaint** | Pruning to `window + 8` is an unjustified heuristic. |
| **Was true?** | Yes — `+8` was a stand-size buffer, not derived from local field stats. |
| **Status** | **Closed in code:** history is pruned to exactly `window` (`OrbitPredictor.observe`). `window` itself remains a **stand size** (how much history we keep), not a PASS threshold. |
| **Still not claimed** | Optimal memory length for every workload. |

## R2 — Resonance decay sensitivity

| | |
|---|---|
| **Complaint** | `decay_hold_from_resonance` may be too aggressive/weak vs cosine distribution; no sensitivity analysis. |
| **Was true?** | Yes — formula is local (`hold ≈ clip(R)`), but no sweep was published. |
| **Status** | **Closed as documentation + script:** `analysis/sensitivity_resonance_decay.py` reports how mean hit / field nnz move under scaled cosine regimes. Output is diagnostic (tables), **not** a magic PASS bar. |
| **Still not claimed** | One universal decay schedule for all MoE models. |

## R3 — Synthetic “orbital” structure vs live DeepSeek gate

| | |
|---|---|
| **Complaint** | Method assumes `h` has structure the predictor can learn; live gate may be much weaker than synthetic. |
| **Was true?** | Partially — Tier **S** stream is learnable by design. That must not be sold as live proof. |
| **Status** | **Closed as evidence separation** in [EVIDENCE_TIERS.md](EVIDENCE_TIERS.md): Tier S ≠ Tier L. Live Tier L (v36) already reports **late learning↑ False** — we do **not** claim rising gate-hit on that lean run. Prefetch claim on that run is **miss-wait**, not “orbit = router.” |
| **Still open (research)** | Stronger live predictor–gate alignment on larger stands / more tasks. That is future work, not a hidden PASS. |

## What we will never do to “silence” auditors

- Relabel Tier S plots as DeepSeek live results.
- Invent rising late-hit if the lean lab said False.
- Add magic numeric PASS margins to bury R3.
