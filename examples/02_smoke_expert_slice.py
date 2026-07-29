#!/usr/bin/env python3
"""
Example 02 — load one DeepSeek-V2-Lite expert slice + hit + sleep/evict.

Requires local Hugging Face cache (or network) for:
  deepseek-ai/DeepSeek-V2-Lite-Chat

Verified behavior (lab report 2026-07-24): expert tensors load as a slice;
evict_below_mean reduces resident bytes (e.g. ~51MB → ~17MB in that run).

This script does NOT redistribute model weights.
Respect DeepSeek / Hugging Face model license when downloading.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub import try_to_load_from_cache

from moe_orbit_prefetch import DynamicExpertStore, OrbitPredictor

MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite-Chat"
LAYER = 1
EXPERT_ID = 0


def main() -> int:
    idx = try_to_load_from_cache(MODEL_ID, "model.safetensors.index.json")
    if not idx or not Path(idx).exists():
        print(
            "SKIP: index not in HF cache.\n"
            f"  huggingface-cli download {MODEL_ID} model.safetensors.index.json\n"
            "  (and at least the shard that contains layer-1 expert 0)\n"
            "Then re-run. We do not ship weights in this repository."
        )
        return 0

    store = DynamicExpertStore.from_index_file(
        MODEL_ID, idx, allow_hub_download=False
    )
    eids = store.list_expert_ids(LAYER)
    print(f"model={MODEL_ID} layer={LAYER} n_experts={len(eids)}")
    if EXPERT_ID not in eids:
        print(f"FAIL: expert {EXPERT_ID} not in index")
        return 2

    shard = store.shard_for_expert(LAYER, EXPERT_ID)
    st = store.local_shard_status(shard or "")
    print(f"shard={shard} present={st.get('present')} bytes={st.get('bytes')}")
    if not st.get("present"):
        print(
            "SKIP: expert shard not fully cached locally "
            "(set allow_hub_download=True only if you accept HF download + license)."
        )
        return 0

    pack = store.get_expert(LAYER, EXPERT_ID)
    print(f"loaded keys={sorted(pack.keys())} resident={store.resident_bytes()}")

    # hit path
    _ = store.get_expert(LAYER, EXPERT_ID)
    print(f"after second get: hits={store.n_hits} misses={store.n_misses}")

    # deposit-like mass then sleep neighbors
    store.s_env[(LAYER, EXPERT_ID)] = 10.0
    store.s_env[(LAYER, 1)] = 0.1
    store.s_env[(LAYER, 2)] = 0.1
    # ensure neighbors loaded so evict has something to drop
    for eid in (1, 2):
        if eid in eids:
            try:
                store.get_expert(LAYER, eid)
            except Exception as e:
                print(f"warn load expert {eid}: {e}")
    before = store.resident_bytes()
    dropped = store.evict_below_mean()
    after = store.resident_bytes()
    print(f"evict dropped={dropped} resident {before}→{after} n_hot={len(store.hot)}")

    # orbit API smoke (no claim it matches live gate)
    h = __import__("numpy").zeros(64, dtype="float64")
    h[0] = 1.0
    op = OrbitPredictor(n_experts=len(eids) or 64, top_k=6)
    print(f"orbit predict sample={op.predict(h)}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
