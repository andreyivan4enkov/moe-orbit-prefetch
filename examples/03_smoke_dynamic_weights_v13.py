#!/usr/bin/env python3
"""
dynamic_weight_orbit_v13 — шаг за шагом:

  A) process_gate: ждать чужие прогоны
  B) LOCAL: аудит кэша + smoke динамической загрузки на R1 (полный локальный)
  C) LOCAL: index V2-Lite-Chat + статус шарда expert0 (без сети)
  D) если шарда нет — точечная докачка ТОЛЬКО model-00001-of-000004
  E) smoke get_expert(layer=1, id=0) + hit + evict_below_mean

Эмерджентные гейты: wins>losses / булевы факты наличия; без magic PASS-констант.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

VERSION_DIR = Path(__file__).resolve().parent
LOGS_DIR = VERSION_DIR / "logs"
REPORTS_DIR = VERSION_DIR / "reports"
BENCHMARK = "smoke_dynamic_weights_v13"
LOGS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

ROOT = VERSION_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

from moe_orbit_prefetch.dynamic_expert_store import DynamicExpertStore  # noqa: E402
from moe_orbit_prefetch.process_gate import wait_until_idle  # noqa: E402

V2_CHAT = "deepseek-ai/DeepSeek-V2-Lite-Chat"
V2_BASE = "deepseek-ai/DeepSeek-V2-Lite"
R1 = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
SHARD1 = "model-00001-of-000004.safetensors"


class Tee(TextIO):
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            st.flush()


def chk(checks: list[str], name: str, cond: bool, detail: str) -> bool:
    checks.append(f"- [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    return cond


def smoke_r1_local(checks: list[str]) -> bool:
    """Доказать механизм: открыть локальный шард R1 и вытащить один тензор."""
    from huggingface_hub import try_to_load_from_cache, hf_hub_download
    from safetensors import safe_open

    print("\n=== B) LOCAL smoke on R1-Distill (already cached) ===")
    idx_p = try_to_load_from_cache(R1, "model.safetensors.index.json")
    if not idx_p:
        idx_p = hf_hub_download(R1, "model.safetensors.index.json")
    idx = json.loads(Path(idx_p).read_text())
    emb_name = "model.embed_tokens.weight"
    shard = idx["weight_map"][emb_name]
    shard_p = try_to_load_from_cache(R1, shard)
    ok_path = bool(shard_p and Path(shard_p).exists() and Path(shard_p).stat().st_size > 1_000_000)
    if not chk(checks, "r1_shard_local", ok_path, f"shard={shard} path={shard_p}"):
        return False
    with safe_open(str(shard_p), framework="pt", device="cpu") as f:
        t = f.get_tensor(emb_name)
    nbytes = int(t.numel() * t.element_size())
    return chk(checks, "r1_dynamic_tensor", t.ndim == 2 and nbytes > 0, f"shape={tuple(t.shape)} bytes={nbytes}")


def resolve_v2_index() -> tuple[str, Path] | None:
    from huggingface_hub import try_to_load_from_cache

    for mid in (V2_CHAT, V2_BASE):
        p = try_to_load_from_cache(mid, "model.safetensors.index.json")
        if p and Path(p).exists():
            return mid, Path(p)
    return None


def step_c_local_v2(checks: list[str]) -> tuple[DynamicExpertStore | None, dict]:
    print("\n=== C) LOCAL audit V2 MoE index (no download) ===")
    found = resolve_v2_index()
    if not found:
        chk(checks, "v2_index_local", False, "index not in cache")
        return None, {}
    mid, idx_path = found
    store = DynamicExpertStore.from_index_file(mid, idx_path, allow_hub_download=False)
    eids = store.list_expert_ids(1)
    shard = store.shard_for_expert(1, 0)
    st = store.local_shard_status(shard or "")
    chk(checks, "v2_index_local", True, f"model={mid} experts_layer1={len(eids)}")
    chk(
        checks,
        "v2_expert0_mapped",
        shard is not None and len(store.expert_tensor_names(1, 0)) > 0,
        f"shard={shard} n_tensors={len(store.expert_tensor_names(1, 0))}",
    )
    present = bool(st.get("present"))
    chk(
        checks,
        "v2_shard1_local_complete",
        present,
        f"bytes={st.get('bytes')} path={st.get('path')}",
    )
    return store, {"model_id": mid, "shard": shard, "shard_status": st, "n_experts_l1": len(eids)}


def step_d_download(store: DynamicExpertStore, shard: str, checks: list[str]) -> bool:
    print(f"\n=== D) DOWNLOAD single shard {shard} (~8.6GB) ===")
    wait_until_idle(reason="before shard download")
    store.allow_hub_download = True
    try:
        path = store.resolve_shard_path(shard)
    except Exception as e:
        return chk(checks, "download_shard1", False, f"err={type(e).__name__}: {e}")
    ok = bool(path and path.exists() and path.stat().st_size > 1_000_000_000)
    return chk(
        checks,
        "download_shard1",
        ok,
        f"path={path} bytes={path.stat().st_size if path else 0}",
    )


def step_e_expert_smoke(store: DynamicExpertStore, checks: list[str]) -> bool:
    print("\n=== E) get_expert(1,0) + second hit + sleep evict ===")
    wait_until_idle(reason="before expert smoke")
    store.allow_hub_download = False  # must be local now
    try:
        pack = store.get_expert(1, 0)
    except Exception as e:
        return chk(checks, "load_expert0", False, f"{type(e).__name__}: {e}")
    ok1 = chk(
        checks,
        "load_expert0",
        len(pack) > 0 and store.resident_bytes() > 0,
        f"keys={list(pack.keys())} resident={store.resident_bytes()}",
    )
    # second call = hit
    before_hits = store.n_hits
    _ = store.get_expert(1, 0)
    ok2 = chk(checks, "hot_hit", store.n_hits == before_hits + 1, f"hits={store.n_hits}")
    # load another expert to create field diversity if same shard
    try:
        store.get_expert(1, 1)
        store.get_expert(1, 2)
    except Exception as e:
        print(f"  (optional experts 1/2 skip: {e})")
    # boost s_env for 0 so mean-evict drops colder ones
    store.s_env[(1, 0)] = store.s_env.get((1, 0), 0.0) + 10.0
    res_before = store.resident_bytes()
    dropped = store.evict_below_mean()
    res_after = store.resident_bytes()
    ok3 = chk(
        checks,
        "evict_sleep",
        res_after < res_before or len(dropped) > 0,
        f"dropped={dropped} resident {res_before}→{res_after} hot={store.n_hot()}",
    )
    return ok1 and ok2 and ok3


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{BENCHMARK}_{stamp}.log"
    report_path = REPORTS_DIR / f"{BENCHMARK}_{stamp}_report.md"
    json_path = REPORTS_DIR / f"{BENCHMARK}_{stamp}_results.json"

    log_f = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_f)

    print(f"=== {BENCHMARK} === phase 5a dynamic weights", flush=True)
    wait_until_idle(reason="start smoke")
    print("process_gate cleared", flush=True)

    checks: list[str] = []
    meta: dict = {}

    ok_r1 = smoke_r1_local(checks)
    store, meta_c = step_c_local_v2(checks)
    meta["local_v2"] = meta_c

    downloaded = False
    expert_ok = False
    if store is not None:
        shard = meta_c.get("shard") or SHARD1
        st = meta_c.get("shard_status") or {}
        if not st.get("present"):
            print("\nLocal shard incomplete → pump ONE shard only (user-approved).")
            downloaded = step_d_download(store, shard, checks)
            # refresh status
            st2 = store.local_shard_status(shard)
            meta["after_download"] = st2
        else:
            chk(checks, "download_shard1", True, "skipped — already local")
            downloaded = True

        if st.get("present") or downloaded:
            # recreate store with download disabled for load path purity
            mid = meta_c["model_id"]
            store2 = DynamicExpertStore.from_index_file(mid, store.index_path, allow_hub_download=False)
            expert_ok = step_e_expert_smoke(store2, checks)
            meta["store_stats"] = store2.stats()
        else:
            chk(checks, "load_expert0", False, "no shard after download attempt")
    else:
        chk(checks, "download_shard1", False, "no index — cannot download targeted shard")

    all_fail = [c for c in checks if c.startswith("- [FAIL]")]
    # PASS if R1 local smoke ok AND (expert smoke ok OR we documented download failure)
    if ok_r1 and expert_ok:
        v = "PASS_PHASE5A_DYNAMIC"
    elif ok_r1 and store is not None and not expert_ok:
        v = "FAIL_PHASE5A_EXPERT_LOAD"
    elif ok_r1:
        v = "PARTIAL_LOCAL_ONLY_R1"
    else:
        v = "FAIL_PHASE5A"

    print(f"\nVERDICT: {v}")
    for c in checks:
        print(c)

    report_path.write_text(
        "\n".join(
            [
                f"# {BENCHMARK}",
                "",
                f"**verdict:** {v}",
                "",
                "## Checklist",
                "",
                *checks,
                "",
                "## Plain reading",
                "",
                "Сначала локально (R1 + index V2). Потом при необходимости один шард ~8.6GB.",
                "Эксперт грузится куском, не вся модель. Sleep = evict по mean S_env.",
                "",
                f"stats={json.dumps(meta.get('store_stats', {}), ensure_ascii=False)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps({"verdict": v, "checks": checks, "meta": meta}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"report={report_path}")
    log_f.close()
    sys.stdout = sys.__stdout__
    if v not in ("PASS_PHASE5A_DYNAMIC", "PARTIAL_LOCAL_ONLY_R1"):
        raise SystemExit(2)
    if v == "PARTIAL_LOCAL_ONLY_R1":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
