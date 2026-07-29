# Changelog

## 0.5.5 — 2026-07-29

- Added public `examples/05_gigachat_store_smoke.py` for MacBook-checkable GigaChat store validation
- Verified editable install/import path and expanded `LOCAL_VALIDATION_20260729.md`
- Ignored generated `examples/logs/` and `examples/reports/` so honest local runs do not dirty the tree

## 0.5.4 — 2026-07-29

- Added `docs/RESEARCH_INTENT.md` to separate **objective observation** from **interpretive hypothesis**
- Clarified that “alternative architecture gives non-noise signal” is evidence for further study, **not** proof of a grand intelligence theory

## 0.5.3 — 2026-07-29

- Tightened public hardware honesty: exact author lab now documented as **MacBook Pro 2019 / Intel i9 / 16 GB RAM / Radeon 4 GB**
- Added explicit note about **thermal throttling** and why live numbers should be read as constrained-machine evidence

## 0.5.2 — 2026-07-29

- Local no-synthetic validation recorded in `docs/LOCAL_VALIDATION_20260729.md`
- `SparseDeepseekRuntime`: shared modeled-prefetch state (`_cur_tok_id`, `last_h_vec`) now guarded by `_state_lock`
- Fixed `examples/03_smoke_dynamic_weights_v13.py` so it runs from the repository clone layout
- Added `docs/AUTHORSHIP.md` and linked authorship / AI-assisted workflow from public docs

## 0.5.1 — 2026-07-29

- Document **DeepSeek-family** scope including **GigaChat** lab evidence (`docs/SUPPORTED_MODELS.md`)
- Publish sanitized GigaChat results: v21 / v32 / v34 under `results/gigachat_*`
- README / MODEL_CARD / WHAT_WE_CLAIM: not “DeepSeek-only”, not “any MoE”

## 0.5.0 — 2026-07-29

- GitHub / ML-industry packaging: `MODEL_CARD.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `docs/LAB_SCOPE.md`, CI workflow
- README: badges, honest quick facts, laptop-lab limits (no fake 100-task claim)
- **Auditor triage:** `docs/AUDITOR_ISSUES.md`
- Expert store: load coalesce, `drop_expert`, cold-first evict, CUDA direct refused by default
- Prefetch worker: error counters + optional fail-fast; cancel drains epoch/queue
- `emerges_greater` skips NaN/Inf; `tests/test_store_and_metrics.py`
- Evidence map / risks / sensitivity (from 0.4.1 line)

## 0.4.1 — 2026-07-29

- **Evidence map:** `docs/EVIDENCE_TIERS.md` (Tier L live vs Tier S synthetic) — fixes auditor confusion
- **Risks status:** `docs/RISKS.md` (window+8 removed; sensitivity script; R3 honesty)
- README leads with **live** results + how to re-run on real DeepSeek
- v36 report no longer claims harness is “not in this repo”
- `OrbitPredictor` prunes history to exact `window`
- `analysis/sensitivity_resonance_decay.py` Tier-S diagnostic

## 0.4.0 — 2026-07-29

- **Analysis pack** for reviewers: 200-step trajectory JSON, PNG figures, gates vs classical baselines
- `docs/ARCHITECTURE.md` with system mermaid
- `analysis/REPORT.md` + regenerate scripts
- README previously led with analyzable artifacts (no weights required)

## 0.3.0 — 2026-07-29

- Apache-2.0 + ATTRIBUTION.md; MATH.md; SOURCE_MANIFEST.md

## 0.2.0 — 2026-07-29

- Full sparse runtime + chat + benches

## 0.1.0 — 2026-07-29

- Initial OrbitPredictor + DynamicExpertStore

