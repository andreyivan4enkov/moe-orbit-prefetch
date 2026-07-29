# Local validation report — 2026-07-29

This report records **only checks run locally in this session** on cached real-model weights.  
No synthetic analysis scripts were used for the verdict below.

## Environment

- Repository: `moe-orbit-prefetch`
- Machine: MacBook Pro (2019), Intel Core i9, 16 GB RAM, Radeon 4 GB
- Python: `/Users/corpuscul/Desktop/theory_megaattractor/.venv/bin/python`
- Weight sources: local Hugging Face cache

## Changes validated in this session

1. `SparseDeepseekRuntime`: shared modeled-prefetch state is now updated/snapshotted under `_state_lock`
   (`_cur_tok_id`, `last_h_vec`) to reduce races between main path and prefetch path.
2. `examples/03_smoke_dynamic_weights_v13.py`: fixed `sys.path` bootstrap so the smoke script runs from a clean clone.
3. Added `examples/05_gigachat_store_smoke.py` so GigaChat store-path can be checked by a public one-command example instead of an ad-hoc snippet.
4. Added `docs/AUTHORSHIP.md` and linked it from public docs to state the AI-assisted implementation workflow and the maintainer's role honestly.

## Objective checks run

### A. Unit tests

Command:

```bash
/Users/corpuscul/Desktop/theory_megaattractor/.venv/bin/python -m pytest tests/ -q
```

Result: **PASS** (`5 passed`)

### A2. Editable install / import smoke

Commands:

```bash
/Users/corpuscul/Desktop/theory_megaattractor/.venv/bin/python -m pip install -e .
/Users/corpuscul/Desktop/theory_megaattractor/.venv/bin/python -c "import moe_orbit_prefetch as m; print(m.__version__)"
```

Result: **PASS** (`import_ok 0.5.4` at validation time)

### B. DeepSeek live smoke — expert slice

Command:

```bash
/Users/corpuscul/Desktop/theory_megaattractor/.venv/bin/python examples/02_smoke_expert_slice.py
```

Result: **PASS**

Observed:

- `layer=1 n_experts=64`
- expert tensors loaded: `down_proj.weight`, `gate_proj.weight`, `up_proj.weight`
- second `get_expert` produced a hot hit (`hits=1 misses=1`)
- `evict_below_mean` reduced residency `51904512 -> 17301504`

### C. DeepSeek live smoke — dynamic weights v13

Command:

```bash
/Users/corpuscul/Desktop/theory_megaattractor/.venv/bin/python examples/03_smoke_dynamic_weights_v13.py
```

First run: **FAIL** with `ModuleNotFoundError: No module named 'moe_orbit_prefetch'`

Cause:

- wrong bootstrap path: script used `VERSION_DIR.parents[1]` instead of repository root

Fix applied:

- `ROOT = VERSION_DIR.parent`

Second run after fix: **PASS** (`PASS_PHASE5A_DYNAMIC`)

Observed:

- local R1 shard smoke passed
- V2-Lite index found 64 experts on layer 1
- expert0 mapped and loaded
- hot hit works
- sleep/evict works

### D. DeepSeek live smoke — sparse chat

Command:

```bash
/Users/corpuscul/Desktop/theory_megaattractor/.venv/bin/python examples/04_chat_ask.py "Hello" --max-new 16
```

Result: **PASS**

Observed:

- runtime loaded without full `from_pretrained`
- `spine_bytes=2.62GB`
- after generate: `n_hot=86 resident=1487.9MB loads=1469 hits=941 misses=1469 modeled_hit=952`
- reply produced:

```text
Hello! How can I help you today? If you have any questions or need
```

### E. GigaChat live smoke — public store path on 10B

Command:

```bash
/Users/corpuscul/Desktop/theory_megaattractor/.venv/bin/python examples/05_gigachat_store_smoke.py --model-id ai-sage/GigaChat3-10B-A1.8B-bf16
```

Result: **PASS**

Observed:

- model: `ai-sage/GigaChat3-10B-A1.8B-bf16`
- layer 1 has 64 experts
- expert pack keys: `down_proj.weight`, `gate_proj.weight`, `up_proj.weight`
- residency and hot-hit behavior are sane (`hits=1 misses=1`)

### F. GigaChat live smoke — public store path on 20B

Command:

```bash
/Users/corpuscul/Desktop/theory_megaattractor/.venv/bin/python examples/05_gigachat_store_smoke.py --model-id ai-sage/GigaChat-20B-A3B-instruct-v1.5-bf16
```

Result: **SKIP**

Observed:

- `model.safetensors.index.json` is not available as a normal local cache path in this environment
- smoke script correctly exits with a transparent skip instead of pretending success

## What was NOT fully validated in this session

1. **GigaChat-20B** was only revalidated to the level of a transparent **SKIP**: the public smoke correctly reports missing local index instead of fabricating a pass.
2. No long lean code bench was rerun in this session because that is multi-hour on this laptop.
3. No synthetic scripts were used for this report by design.

## Honest verdict

**Positive with limits.**

What is positively revalidated locally now:

- public DeepSeek live path (`02`, `03`, `04`)
- editable install + import path
- dynamic store hit/evict behavior
- current runtime still produces a real sparse-chat answer
- public GigaChat 10B store-path compatibility
- the repaired `03` example now works from a clean clone layout
- a public GigaChat smoke now exists as `examples/05_gigachat_store_smoke.py`

What remains outside this session's green zone:

- GigaChat-20B package-path revalidation
- long lean benchmark reruns
- wider GPU / multi-device validation

So the repository can be updated **honestly** if the release notes say exactly that, and do **not** claim a fresh full revalidation of every historical lab artifact.
