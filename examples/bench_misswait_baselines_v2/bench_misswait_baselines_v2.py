#!/usr/bin/env python3
"""
Short LIVE miss-wait head-to-head v2 on DeepSeek-V2-Lite-Chat (MacBook-lab).

Primary order (hard): orbit → sgd → frequency → none.
If ours errors — STOP before remaining classics.

Also optional cache_sens phase: frequency → none → orbit (orbit last) to
partially probe OS shard-cache confound from v1. Not a full cold-disk control.

Gate: lower miss-wait better → emerges_greater on negated miss series.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

VERSION_DIR = Path(__file__).resolve().parent
EXPECTED = "bench_misswait_baselines_v2"
if VERSION_DIR.name != EXPECTED:
    raise RuntimeError(f"script must live in {EXPECTED}/, got {VERSION_DIR.name}")

LOGS_DIR = VERSION_DIR / "logs"
REPORTS_DIR = VERSION_DIR / "reports"
LOGS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

ROOT = VERSION_DIR.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(VERSION_DIR.parent / "bench_humaneval_lean"))

BENCHMARK = "bench_misswait_baselines_v2"

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

DEFAULT_PROMPTS = [
    "Write a Python function add(a, b) that returns a+b. Output ONLY the function.",
    "Fix this bug and output ONLY the corrected function:\n"
    "def mul(x, y):\n    return x + y  # should multiply\n",
]

MODES_PRIMARY: list[tuple[str, str]] = [
    ("orbit", "ours"),
    ("sgd", "classic"),
    ("frequency", "classic"),
    ("none", "classic"),
]

# Orbit last — partial cache-order check (not true disk-cold).
MODES_CACHE_SENS: list[tuple[str, str]] = [
    ("frequency", "classic"),
    ("none", "classic"),
    ("orbit", "ours"),
]


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


def make_factory(kind: str, n_experts: int, top_k: int, window: int) -> Callable:
    from moe_orbit_prefetch.classic_prefetch_predictors import (
        FrequencyPredictor,
        LruPredictor,
        OnlineSgdPredictor,
        PrevCopyPredictor,
    )
    from moe_orbit_prefetch.orbit_predictor import OrbitPredictor

    def factory():
        if kind == "orbit":
            return OrbitPredictor(n_experts=n_experts, top_k=top_k, window=window)
        if kind == "frequency":
            return FrequencyPredictor(n_experts=n_experts, top_k=top_k, window=window)
        if kind == "lru":
            return LruPredictor(n_experts=n_experts, top_k=top_k, window=window)
        if kind == "prev_copy":
            return PrevCopyPredictor(n_experts=n_experts, top_k=top_k, window=window)
        if kind == "sgd":
            return OnlineSgdPredictor(n_experts=n_experts, top_k=top_k, window=window)
        raise ValueError(kind)

    return factory


def relieve(runtime: Any) -> int:
    dropped = 0
    store = runtime.experts
    if store is not None:
        # Do NOT wrap evict_below_mean in store._lock: drop_expert acquires
        # the same non-reentrant Lock → self-deadlock (MacBook hang).
        dropped += len(store.evict_below_mean())
        if hasattr(runtime, "trim_hot_to_orbit_cap"):
            dropped += int(runtime.trim_hot_to_orbit_cap())
        else:
            runtime._drop_pack_dev_missing()
    if hasattr(runtime, "clear_fp32_cache"):
        runtime.clear_fp32_cache()
    gc.collect()
    return dropped


def run_one(
    runtime: Any,
    tokenizer: Any,
    prompt: str,
    *,
    mode: str,
    max_new: int,
    temperature: float,
) -> dict[str, Any]:
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = f"User: {prompt}\n\nAssistant:"
    input_ids = tokenizer(rendered, return_tensors="pt")["input_ids"]
    use_pref = mode != "none"
    runtime.use_modeled_prefetch = use_pref
    runtime.token_horizon = 4 if use_pref else 1
    runtime.prefetch_horizon = 2 if use_pref else 0
    runtime.keep_hot_during_generate = use_pref
    eos = getattr(tokenizer, "eos_token_id", None)

    def progress(event: dict[str, Any]) -> None:
        phase = event.get("phase")
        if phase == "moe":
            layer = int(event.get("layer") or 0)
            dropped = 0
            if layer % 4 == 0:
                dropped = relieve(runtime)
            print(
                f"    prefill layer={layer} miss_s={runtime.gate_miss_seconds:.1f} "
                f"sleep={dropped} mode={mode}",
                flush=True,
            )
            return
        if phase != "token_done":
            return
        step = int(event.get("step") or 0)
        relieve(runtime)
        if step == 1 or step % 4 == 0:
            print(
                f"    pulse: tok={step}/{max_new} miss={event.get('gate_miss_s')}s "
                f"mode={mode}",
                flush=True,
            )

    prev_cb = runtime.progress_cb
    runtime.progress_cb = progress
    t0 = time.perf_counter()
    try:
        out = runtime.generate(
            input_ids,
            max_new_tokens=max_new,
            temperature=temperature,
            eos_token_id=eos,
        )
        text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True).strip()
    finally:
        runtime.progress_cb = prev_cb
    wall = time.perf_counter() - t0
    st = runtime.stats()
    miss = float(st.get("gate_miss_seconds") or 0.0)
    return {
        "mode": mode,
        "prompt_head": prompt[:80].replace("\n", " "),
        "reply_head": (text or "")[:120].replace("\n", " "),
        "reply_len": len(text or ""),
        "tokens_out": int(out.shape[-1]),
        "wall_s": round(wall, 3),
        "miss_wait_s": miss,
        "gate_hit": int(st.get("n_gate_pack_hit") or 0),
        "gate_miss": int(st.get("n_gate_pack_miss") or 0),
        "modeled_hit": int(st.get("n_modeled_hit") or 0),
        "n_prefetch_failures": int(st.get("n_prefetch_failures") or 0),
        "ok": bool(text) and miss == miss,  # finite + non-empty
    }


def run_phase(
    *,
    phase_name: str,
    modes: list[tuple[str, str]],
    runtime: Any,
    tokenizer: Any,
    prompts: list[str],
    n_experts: int,
    top_k: int,
    args: argparse.Namespace,
    emerges_greater: Callable,
    stop_on_orbit_fail: bool,
) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    stop_reason: str | None = None
    print(f"\n===== PHASE {phase_name} =====", flush=True)
    print("order: " + " → ".join(m for m, _ in modes), flush=True)

    for mode, family in modes:
        print(f"\n--- MODE {mode} ({family}) ---", flush=True)
        runtime.clear_expert_residency()
        if mode == "none":
            runtime.predictors = {}
            runtime.use_modeled_prefetch = False
        else:
            runtime.install_predictors(
                make_factory(mode, n_experts, top_k, args.window)
            )
            runtime.use_modeled_prefetch = True

        rows: list[dict[str, Any]] = []
        for i, prompt in enumerate(prompts):
            print(f"  prompt[{i}] head={prompt[:60]!r}…", flush=True)
            runtime.clear_expert_residency()
            try:
                row = run_one(
                    runtime,
                    tokenizer,
                    prompt,
                    mode=mode,
                    max_new=args.max_new,
                    temperature=args.temperature,
                )
            except Exception as e:
                print(f"FAIL mode={mode}: {type(e).__name__}: {e}", flush=True)
                stop_reason = f"{mode}_exception:{type(e).__name__}"
                rows.append(
                    {
                        "mode": mode,
                        "ok": False,
                        "error": f"{type(e).__name__}: {e}",
                        "miss_wait_s": float("nan"),
                    }
                )
                by_mode[mode] = rows
                break
            print(
                f"  → ok={row['ok']} miss={row['miss_wait_s']:.2f}s "
                f"wall={row['wall_s']:.1f}s reply_len={row['reply_len']}",
                flush=True,
            )
            rows.append(row)
            if stop_on_orbit_fail and mode == "orbit" and not row["ok"]:
                stop_reason = "orbit_failed_ok_gate"
                break
        by_mode[mode] = rows
        if stop_reason:
            print(f"STOP: {stop_reason}", flush=True)
            break
        if stop_on_orbit_fail and mode == "orbit":
            if not rows or not all(r.get("ok") for r in rows):
                stop_reason = "orbit_incomplete"
                print(f"STOP: {stop_reason}", flush=True)
                break

    ours_rows = by_mode.get("orbit") or []
    ours_miss = [float(r["miss_wait_s"]) for r in ours_rows if r.get("ok")]
    comparisons: dict[str, Any] = {}
    for mode, family in modes:
        if mode == "orbit" or family != "classic":
            continue
        other = by_mode.get(mode) or []
        if len(other) != len(ours_rows) or not ours_miss:
            comparisons[mode] = {
                "status": "skipped",
                "reason": stop_reason or "incomplete",
            }
            continue
        other_miss = [float(r["miss_wait_s"]) for r in other]
        better, votes = emerges_greater(
            [-v for v in ours_miss], [-v for v in other_miss]
        )
        comparisons[mode] = {
            "status": "compared",
            "ours_better_miss": better,
            "votes": votes,
            "mean_miss_ours": sum(ours_miss) / len(ours_miss),
            "mean_miss_other": sum(other_miss) / len(other_miss),
            "per_prompt_ours": ours_miss,
            "per_prompt_other": other_miss,
        }
    return {
        "phase": phase_name,
        "modes": modes,
        "by_mode": by_mode,
        "comparisons": comparisons,
        "stop_reason": stop_reason,
    }


def summarize_phase(phase: dict[str, Any]) -> tuple[str, list[str]]:
    lines: list[str] = [
        f"### Phase `{phase['phase']}`",
        "",
        "order: " + " → ".join(m for m, _ in phase["modes"]),
        f"stop_reason: `{phase.get('stop_reason') or 'none'}`",
        "",
        "| mode | family | mean miss-s | n ok |",
        "|---|---|---:|---:|",
    ]
    by_mode = phase["by_mode"]
    for mode, family in phase["modes"]:
        rows = by_mode.get(mode) or []
        oks = [r for r in rows if r.get("ok")]
        if not oks:
            lines.append(f"| {mode} | {family} | — | 0 |")
            continue
        mean_m = sum(float(r["miss_wait_s"]) for r in oks) / len(oks)
        lines.append(f"| {mode} | {family} | {mean_m:.2f} | {len(oks)} |")
    lines += ["", "Ours vs classic (lower miss better):", ""]
    any_win = False
    any_loss = False
    compared = False
    for mode, cmp_ in phase["comparisons"].items():
        if cmp_.get("status") != "compared":
            lines.append(f"- `{mode}`: skipped ({cmp_.get('reason')})")
            continue
        compared = True
        v = cmp_["votes"]
        flag = "PASS" if cmp_["ours_better_miss"] else "FAIL"
        any_win |= bool(cmp_["ours_better_miss"])
        any_loss |= not bool(cmp_["ours_better_miss"])
        lines.append(
            f"- `{mode}`: **{flag}** wins={v['wins']} losses={v['losses']} "
            f"mean_ours={cmp_['mean_miss_ours']:.2f}s "
            f"mean_{mode}={cmp_['mean_miss_other']:.2f}s"
        )
    if phase.get("stop_reason"):
        verdict = f"STOPPED ({phase['stop_reason']})"
    elif not compared:
        verdict = "INCOMPLETE"
    elif any_win and not any_loss:
        verdict = "PASS_OURS_BEATS_ALL_COMPARED_CLASSICS"
    elif any_win and any_loss:
        verdict = "MIXED"
    else:
        verdict = "FAIL_OURS_NOT_BETTER_ON_COMPARED"
    lines += ["", f"Phase verdict: `{verdict}`", ""]
    return verdict, lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--n-prompts", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument(
        "--phase",
        choices=("primary", "cache_sens", "both"),
        default="both",
        help="primary=orbit first+SGD; cache_sens=orbit last; both=sequential",
    )
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{BENCHMARK}_{stamp}.log"
    report_path = REPORTS_DIR / f"{BENCHMARK}_{stamp}_report.md"
    json_path = REPORTS_DIR / f"{BENCHMARK}_{stamp}_results.json"

    log_f = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)

    print(f"=== {BENCHMARK} ===")
    print(f"log={log_path}")
    print("hardware note: MacBook Pro 2019 Intel i9 / 16GB — thermal throttling expected")
    print(f"phase={args.phase}")

    from moe_orbit_prefetch.emergent_metrics import emerges_greater
    from moe_orbit_prefetch.process_gate import wait_until_idle
    from transformers import AutoTokenizer
    from bench_humaneval_lean import LeanDeepseekRuntime

    wait_until_idle()
    print("Loading LEAN sparse DeepSeek-V2-Lite-Chat…", flush=True)
    runtime = LeanDeepseekRuntime.load_lean(
        use_modeled_prefetch=True,
        prefetch_horizon=2,
        token_horizon=4,
    )
    runtime.progress_every_n_layers = 1
    n_experts = int(runtime.cfg["n_routed_experts"])
    top_k = int(runtime.cfg["num_experts_per_tok"])
    runtime.pack_dev_cap = max(int(runtime.pack_dev_cap), top_k * 8)
    tokenizer = AutoTokenizer.from_pretrained(runtime.model_id, trust_remote_code=True)
    print(
        f"model={runtime.model_id} n_experts={n_experts} top_k={top_k} "
        f"spine={runtime.spine_bytes/1e9:.2f}GB pack_cap={runtime.pack_dev_cap}",
        flush=True,
    )

    prompts = DEFAULT_PROMPTS[: max(1, args.n_prompts)]
    phases_to_run: list[tuple[str, list[tuple[str, str]], bool]] = []
    if args.phase in ("primary", "both"):
        phases_to_run.append(("primary", MODES_PRIMARY, True))
    if args.phase in ("cache_sens", "both"):
        phases_to_run.append(("cache_sens", MODES_CACHE_SENS, False))

    phase_results: list[dict[str, Any]] = []
    for name, modes, stop_orbit in phases_to_run:
        phase_results.append(
            run_phase(
                phase_name=name,
                modes=modes,
                runtime=runtime,
                tokenizer=tokenizer,
                prompts=prompts,
                n_experts=n_experts,
                top_k=top_k,
                args=args,
                emerges_greater=emerges_greater,
                stop_on_orbit_fail=stop_orbit,
            )
        )
        if name == "primary" and phase_results[-1].get("stop_reason"):
            print("Primary ours failed — skipping later phases", flush=True)
            break

    payload = {
        "benchmark": BENCHMARK,
        "stamp": stamp,
        "hardware": "MacBook Pro 2019 Intel i9 16GB Radeon4GB (thermal throttling)",
        "max_new": args.max_new,
        "n_prompts": len(prompts),
        "phase_arg": args.phase,
        "phases": phase_results,
        "ontology_checklist": {
            "predict_from_embeddings_topology_for_ours": True,
            "classic_baselines_are_not_ours": True,
            "includes_online_sgd": True,
            "no_oracle_deposit": True,
            "ours_first_on_primary_phase": True,
            "cache_sens_is_partial_not_disk_cold": True,
            "emergent_wins_gt_losses": True,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# {BENCHMARK}",
        "",
        f"stamp: `{stamp}`",
        "hardware: MacBook Pro 2019 Intel i9 / 16 GB (throttling)",
        f"max_new={args.max_new} n_prompts={len(prompts)} phase={args.phase}",
        "",
        "## Meaning",
        "",
        "Primary: ours first, then online SGD, frequency, none.",
        "cache_sens: classics first, orbit last (partial OS-cache check only).",
        "",
    ]
    verdicts: list[str] = []
    for ph in phase_results:
        v, chunk = summarize_phase(ph)
        verdicts.append(f"{ph['phase']}:{v}")
        lines.extend(chunk)

    overall = "; ".join(verdicts) if verdicts else "INCOMPLETE"
    lines += [
        f"## Overall: `{overall}`",
        "",
        "Not a multi-GPU HumanEval. Honest laptop diagnostic only.",
        f"JSON: `{json_path.name}`",
        f"Log: `{log_path.name}`",
        "",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    print(f"report={report_path}", flush=True)

    if any("STOPPED" in v or "INCOMPLETE" in v for v in verdicts):
        return 2
    if any("FAIL" in v for v in verdicts) and not any("PASS" in v for v in verdicts):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
