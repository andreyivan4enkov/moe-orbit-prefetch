# Contributing

Thanks for considering a contribution. This project follows common GitHub / open-ML norms.

## Scope

1. Keep **Object A** (MoE expert orbit / residency). Object B (chat topology / RLM) stays out.
2. Do not commit model weights, secrets, or large binaries.
3. If you change equations, update [docs/MATH.md](docs/MATH.md) in the same PR.
4. New **claims** need a lab artifact under `results/` — no invented PASS.
5. Prefer **ours → classic** order in benches; use emergent `wins > losses` gates (no magic margins).
6. Label Tier **S** (synthetic) vs Tier **L** (live) clearly — [docs/EVIDENCE_TIERS.md](docs/EVIDENCE_TIERS.md).
7. Do not misrepresent authorship: see [docs/AUTHORSHIP.md](docs/AUTHORSHIP.md) for the project's AI-assisted workflow and maintainer role.

## Dev setup

```bash
pip install -e ".[analysis]"
pip install pytest
pytest tests/ -q
```

CI runs the same unit tests on every PR (no HF weights required).

## Larger benches

Author lab is laptop-bound ([docs/LAB_SCOPE.md](docs/LAB_SCOPE.md)). GPU / 100-task artifacts are welcome as PRs with honest reports.

## License

Contributions are under **Apache-2.0**. Keep `LICENSE` / `NOTICE`. Substantial products: see [ATTRIBUTION.md](ATTRIBUTION.md).

## Conduct

[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
