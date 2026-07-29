#!/usr/bin/env python3
"""
Example 05 — load one GigaChat expert slice + hit + sleep/evict.

Requires local Hugging Face cache for one of:
  - ai-sage/GigaChat3-10B-A1.8B-bf16
  - ai-sage/GigaChat-20B-A3B-instruct-v1.5-bf16

This smoke validates the public DeepSeek-family store path on GigaChat-style MoE
without loading the full model through from_pretrained.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub import try_to_load_from_cache  # noqa: E402

from moe_orbit_prefetch import DynamicExpertStore  # noqa: E402

MODEL_CHOICES = [
    "ai-sage/GigaChat3-10B-A1.8B-bf16",
    "ai-sage/GigaChat-20B-A3B-instruct-v1.5-bf16",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_CHOICES[0], choices=MODEL_CHOICES)
    ap.add_argument("--layer", type=int, default=1)
    args = ap.parse_args()

    idx = try_to_load_from_cache(args.model_id, "model.safetensors.index.json")
    if not isinstance(idx, str) or not Path(idx).exists():
        print(
            "SKIP: model.safetensors.index.json not available in local HF cache.\n"
            f"  model={args.model_id}\n"
            "We do not download weights automatically in this smoke."
        )
        return 0

    store = DynamicExpertStore.from_index_file(args.model_id, idx, allow_hub_download=False)
    layer = int(args.layer)
    eids = store.list_expert_ids(layer)
    if not eids:
        print(f"SKIP: no routed experts found on layer={layer}")
        return 0

    eid = eids[0]
    shard = store.shard_for_expert(layer, eid)
    st = store.local_shard_status(shard or "")
    print(
        f"model={args.model_id} layer={layer} n_experts={len(eids)} "
        f"shard={shard} present={st.get('present')}"
    )
    if not st.get("present"):
        print("SKIP: target expert shard is not fully cached locally.")
        return 0

    pack = store.get_expert(layer, eid)
    print(f"loaded keys={sorted(pack.keys())} resident={store.resident_bytes()}")
    _ = store.get_expert(layer, eid)
    print(f"after second get: hits={store.n_hits} misses={store.n_misses}")

    store.s_env[(layer, eid)] = 10.0
    for cold in eids[1:3]:
        try:
            store.get_expert(layer, cold)
            store.s_env[(layer, cold)] = 0.1
        except Exception as e:
            print(f"warn load expert {cold}: {e}")
    before = store.resident_bytes()
    dropped = store.evict_below_mean()
    after = store.resident_bytes()
    print(f"evict dropped={dropped} resident {before}→{after} n_hot={store.n_hot()}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
