# Lab scope & compute policy

## Author compute

Tier **L** evidence in this repository was produced on a **2019 MacBook Pro**. That is intentional: Object A exists to keep MoE residency small enough for constrained machines.

Current public hardware description for the author's lab:

- Apple **MacBook Pro (2019)**
- **Intel Core i9**
- **16 GB RAM**
- discrete **Radeon GPU with 4 GB VRAM**

Important practical limit:

- this machine **thermally throttles hard** under sustained heavy inference / simulation load;
- even moderate background load can reduce throughput;
- therefore long live benches here should be read as **constrained-machine evidence**, not as stable datacenter-style performance numbers.

## What we publish

| Stand | Status |
|---|---|
| Expert-slice smoke (v13) | Published |
| Lean HumanEval + QuixBugs subset (v36) | Published (hours on CPU) |
| Full HumanEval (100+), multi-model, GPU sweep | **Not claimed** — requires external GPU / cloud |

Industry-aligned stance (same as many academic code releases):

1. Ship **reproducible scripts** for the published stand.
2. Document hardware limits honestly.
3. Do not invent larger scoreboards without artifacts.

## Reviewer expectations

Asking for a 100-task GPU suite is reasonable for a *future paper*, not a blocker for reviewing this prototype **if** Tier L lean results and source are clear. See [MODEL_CARD.md](../MODEL_CARD.md) and [EVIDENCE_TIERS.md](EVIDENCE_TIERS.md).

## Community contributions welcome

PRs that add GPU / larger-bench artifacts under `results/` with the same honesty rules (ours → classic, emergent gates, no magic PASS margins) are encouraged.
