# Local validation report — 2026-07-29

This report records **only checks run locally in this session** on cached real-model weights.  
No synthetic analysis scripts were used for the verdict below.

## Environment

- Repository: `moe-orbit-prefetch`
- Machine: author MacBook-class laptop
- Python: `/Users/corpuscul/Desktop/theory_megaattractor/.venv/bin/python`
- Weight sources: local Hugging Face cache

## Changes validated in this session

1. `SparseDeepseekRuntime`: shared modeled-prefetch state is now updated/snapshotted under `_state_lock`
   (`_cur_tok_id`, `last_h_vec`) to reduce races between main path and prefetch path.
2. `examples/03_smoke_dynamic_weights_v13.py`: fixed `sys.path` bootstrap so the smoke script runs from a clean clone.
3. Added `docs/AUTHORSHIP.md` and linked it from public docs to state the AI-assisted implementation workflow and the maintainer's role honestly.

## Objective checks run

### A. Unit tests

Command:

```bash
/Users/corpuscul/Desktop/theory_megaattractor/.venv/bin/python -m pytest tests/ -q
```

Result: **PASS** (`5 passed`)

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

### E. GigaChat live smoke — store path on 10B

Command:

```bash
PYTHONPATH=src /Users/corpuscul/Desktop/theory_megaattractor/.venv/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import try_to_load_from_cache
from moe_orbit_prefetch.dynamic_expert_store import DynamicExpertStore
mid='ai-sage/GigaChat3-10B-A1.8B-bf16'
idx = try_to_load_from_cache(mid, 'model.safetensors.index.json')
st = DynamicExpertStore.from_index_file(mid, Path(idx), allow_hub_download=False)
ids = st.list_expert_ids(1)
pack = st.get_expert(1, ids[0], device='cpu')
_ = st.get_expert(1, ids[0], device='cpu')
print({'model': mid, 'layer': 1, 'n_experts': len(ids), 'keys': sorted(pack.keys()), 'resident_bytes': st.resident_bytes(), 'hits': st.n_hits, 'misses': st.n_misses})
PY
```

Result: **PASS**

Observed:

- model: `ai-sage/GigaChat3-10B-A1.8B-bf16`
- layer 1 has 64 experts
- expert pack keys: `down_proj.weight`, `gate_proj.weight`, `up_proj.weight`
- residency and hot-hit behavior are sane (`hits=1 misses=1`)

## What was NOT fully validated in this session

1. **GigaChat-20B public package smoke** was **not** revalidated by the same direct package-path method in this session:
   local cache has `config.json`, but `model.safetensors.index.json` is not present as a normal local path in this environment.
2. No long lean code bench was rerun in this session because that is multi-hour on this laptop.
3. No synthetic scripts were used for this report by design.

## Honest verdict

**Positive with limits.**

What is positively revalidated locally now:

- public DeepSeek live path (`02`, `03`, `04`)
- dynamic store hit/evict behavior
- current runtime still produces a real sparse-chat answer
- GigaChat 10B store-path compatibility
- the repaired `03` example now works from a clean clone layout

What remains outside this session's green zone:

- GigaChat-20B package-path revalidation
- long lean benchmark reruns
- wider GPU / multi-device validation

So the repository can be updated **honestly** if the release notes say exactly that, and do **not** claim a fresh full revalidation of every historical lab artifact.
