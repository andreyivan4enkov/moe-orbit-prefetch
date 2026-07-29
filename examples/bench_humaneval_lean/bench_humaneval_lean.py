#!/usr/bin/env python3
"""
deepseek_humaneval_orbit_v36_lean — DeepSeek-V2-Lite после OOM v35 (exit 137).

Что чинили: v35 держал весь spine в fp32 (~5.25GB) при ~3GB свободно → ОС убила процесс.
Здесь:
  1. spine хранится в fp16 (~2.6GB), в fp32 кастуются только матрицы активного слоя (LRU);
  2. pack_dev_cap = структурно 2 токена × MoE-слои × top_k (не 512);
  3. sleep жёстче: если свободно < spine ИЛИ swap уже растёт — evict_below_mean;
  4. OrbitPredictor без magic numbers (как в v35);
  5. MAX_NEW=64 — короче ответы под ноут.

Порядок: ours → classic; наша сломалась → СТОП.
"""
from __future__ import annotations

import ast
import gc
import json
import os
import re
import signal
import sys
import textwrap
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TextIO

VERSION_DIR = Path(__file__).resolve().parent
EXPECTED_DIR = "bench_humaneval_lean"
if VERSION_DIR.name != EXPECTED_DIR:
    raise RuntimeError(f"Ожидается {EXPECTED_DIR}/, сейчас {VERSION_DIR.name}")

LOGS_DIR = VERSION_DIR / "logs"
REPORTS_DIR = VERSION_DIR / "reports"
DATA_DIR = VERSION_DIR / "data"
BENCHMARK = "bench_humaneval_lean"
LOGS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

ROOT = VERSION_DIR.parents[2]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import psutil  # noqa: E402
import torch  # noqa: E402
from moe_orbit_prefetch.emergent_metrics import emerges_greater  # noqa: E402
from moe_orbit_prefetch.process_gate import _list_busy_pids  # noqa: E402
from moe_orbit_prefetch.sparse_moe_runtime import MID, SparseDeepseekRuntime  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

STORAGE_DTYPE = torch.float16  # spine на диске/в RAM — компактно
COMPUTE_DTYPE = torch.float32  # Intel считает быстро только так
MAX_NEW = 64
TEMPERATURE = 0.0
GEN_TIMEOUT_S = 2400
HEARTBEAT_EVERY = 8
MAX_REPAIRS = 1
EXEC_TIMEOUT_S = 5
HUMANEVAL_N = int(os.environ.get("HE_N", "3"))
AUDIT_N = int(os.environ.get("AUDIT_N", "3"))


@dataclass
class LeanDeepseekRuntime(SparseDeepseekRuntime):
    """spine в fp16 + ленивый fp32-LRU; pack экспертов остаётся в storage dtype."""

    fp32_budget: int = 0
    _fp32_cache: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _fp32_bytes: int = 0

    def clear_fp32_cache(self) -> None:
        self._fp32_cache.clear()
        self._fp32_bytes = 0

    def trim_hot_to_orbit_cap(self) -> int:
        """Жёсткий sleep: hot ≤ pack_dev_cap (структурный потолок орбиты)."""
        store = self.experts
        if store is None:
            return 0
        cap = max(16, int(self.pack_dev_cap))
        dropped = 0
        with store._lock:
            while store.n_hot() > cap and store.hot:
                # Never evict an expert that another thread is still publishing;
                # that race left waiters blocked on Event with empty hot.
                candidates = [k for k in store.hot if k not in store._loading]
                if not candidates:
                    break
                key = min(candidates, key=lambda k: store.s_env.get(k, 0.0))
                pack = store.hot.pop(key)
                seen: set[int] = set()
                for t in pack.values():
                    if id(t) in seen:
                        continue
                    seen.add(id(t))
                    store.bytes_evicted += int(t.numel() * t.element_size())
                dropped += 1
        if dropped:
            self._drop_pack_dev_missing()
        return dropped

    def _expert_pack_dev(self, layer: int, eid: int, *, count_gate: bool = False) -> dict[str, torch.Tensor]:
        out = super()._expert_pack_dev(layer, eid, count_gate=count_gate)
        # prefetch-worker иначе раздувает hot без потолка (v35/smoke → десятки GB)
        self.trim_hot_to_orbit_cap()
        return out

    def w(self, name: str) -> torch.Tensor:
        tensor = self.spine[name]
        if tensor.device.type != self.device:
            tensor = tensor.to(self.device)
            self.spine[name] = tensor
        if tensor.ndim < 2 or tensor.dtype == COMPUTE_DTYPE:
            return tensor
        cached = self._fp32_cache.get(name)
        if cached is not None:
            self._fp32_cache.move_to_end(name)
            return cached
        promoted = tensor.to(dtype=COMPUTE_DTYPE)
        self._fp32_cache[name] = promoted
        self._fp32_bytes += int(promoted.numel() * promoted.element_size())
        while self._fp32_bytes > self.fp32_budget and len(self._fp32_cache) > 1:
            oldest_key, oldest = next(iter(self._fp32_cache.items()))
            if oldest_key == name:
                break
            self._fp32_cache.pop(oldest_key)
            self._fp32_bytes -= int(oldest.numel() * oldest.element_size())
        return promoted

    @classmethod
    def load_lean(
        cls,
        *,
        model_id: str | None = None,
        use_modeled_prefetch: bool = True,
        prefetch_horizon: int = 2,
        token_horizon: int = 4,
    ) -> "LeanDeepseekRuntime":
        # грузим как SparseDeepseekRuntime (тот же prefetch-worker), затем повышаем класс in-place
        base = SparseDeepseekRuntime.load(
            model_id=model_id or MID,
            dtype=STORAGE_DTYPE,
            use_modeled_prefetch=use_modeled_prefetch,
            prefetch_horizon=prefetch_horizon,
            token_horizon=token_horizon,
        )
        cfg = base.cfg
        top_k = int(cfg.get("num_experts_per_tok", 6))
        per_layer_fp32 = sum(
            int(t.numel() * 4)
            for name, t in base.spine.items()
            if ".layers.1." in name and t.ndim >= 2
        )
        base.__class__ = cls
        base.fp32_budget = max(per_layer_fp32 * 3, 64 * 1024 * 1024)
        base._fp32_cache = OrderedDict()
        base._fp32_bytes = 0
        # pack в fp32: Intel не умеет быстро считать fp16×fp32; spine остаётся fp16 в self.spine
        base.dtype = COMPUTE_DTYPE
        # структурный минимум орбиты под ноут: текущий+след. слой × live+prefetch × top_k
        # (полный moe_layers*top_k*2 раздувает RAM → OOM как v35)
        base.pack_dev_cap = top_k * 2 * 2
        print(
            f"  lean: spine_fp16={base.spine_bytes / 1e9:.2f}GB, "
            f"fp32-LRU={base.fp32_budget / 1e9:.2f}GB, pack_dev_cap={base.pack_dev_cap}",
            flush=True,
        )
        return base  # type: ignore[return-value]


class GenerationTimeout(Exception):
    pass


class ExecTimeout(Exception):
    pass


def _raise_timeout(signum: int, frame: Any) -> None:
    raise GenerationTimeout()


def _raise_exec_timeout(signum: int, frame: Any) -> None:
    raise ExecTimeout()


class Tee(TextIO):
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass(frozen=True)
class Task:
    kind: str
    task_id: str
    prompt_user: str
    entry_point: str
    evaluate: Callable[[str], dict[str, Any]]
    he_prompt: str = ""


def wait_idle() -> None:
    while True:
        busy = [
            (pid, cmd)
            for pid, cmd in _list_busy_pids()
            if EXPECTED_DIR not in cmd and "chat_web" not in cmd
        ]
        if not busy:
            print("process_gate: idle", flush=True)
            return
        print(f"process_gate: ждём {busy[:2]}", flush=True)
        time.sleep(3)


def salvage_code(text: str) -> str:
    fenced = re.search(r"```(?:python)?[ \t]*\n?([\s\S]*?)(?:```|$)", text, re.I)
    if fenced:
        code = textwrap.dedent(fenced.group(1).strip("\n")).rstrip()
    else:
        function = re.search(r"(def\s+\w+\s*\([\s\S]+)", text)
        raw = function.group(1) if function else text
        code = textwrap.dedent(raw.strip("\n")).rstrip()
    if not code:
        return ""
    if re.search(r"\bre\.", code) and "import re" not in code:
        code = "import re\n" + code

    def compiles(candidate: str) -> bool:
        try:
            compile(candidate, "<candidate>", "exec")
            return True
        except SyntaxError:
            pass
        try:
            compile(
                "def _wrapper_():\n" + textwrap.indent(candidate, "    "),
                "<candidate-body>",
                "exec",
            )
            return True
        except SyntaxError:
            return False

    if compiles(code):
        return code
    closing = {"(": ")", "[": "]", "{": "}"}
    lines = code.splitlines()
    while lines:
        head = "\n".join(lines).rstrip()
        if compiles(head):
            return head
        stack: list[str] = []
        quote: str | None = None
        escaped = False
        for char in head:
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in "'\"":
                quote = char
            elif char in closing:
                stack.append(closing[char])
            elif char in ")]}" and stack and stack[-1] == char:
                stack.pop()
        if not quote and stack:
            patched = head + "".join(reversed(stack))
            if compiles(patched):
                return patched
        lines.pop()
    return code


def prompt_header(he_prompt: str) -> str:
    header: list[str] = []
    for line in he_prompt.splitlines():
        if line.startswith("def "):
            break
        header.append(line)
    return "\n".join(header).strip() + "\n"


def indent_body(code: str) -> str:
    lines = code.splitlines()
    if not lines:
        return ""
    non_empty = [line for line in lines if line.strip()]
    base = min((len(line) - len(line.lstrip()) for line in non_empty), default=0)
    out = []
    for line in lines:
        out.append("" if not line.strip() else "    " + line[base:])
    return "\n".join(out)


def build_program(code: str, problem: dict[str, Any]) -> tuple[str, str]:
    entry = problem["entry_point"]
    he_prompt = problem["prompt"]
    test = problem["test"]
    call = f"check({entry})\n"
    if re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", code, re.M):
        return prompt_header(he_prompt) + "\n" + code + "\n\n" + test + "\n" + call, "full_def"
    return he_prompt + indent_body(code) + "\n\n" + test + "\n" + call, "body"


def run_with_exec_timeout(fn: Callable[[], Any], timeout_s: int = EXEC_TIMEOUT_S) -> Any:
    previous = signal.signal(signal.SIGALRM, _raise_exec_timeout)
    signal.alarm(timeout_s)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def evaluate_humaneval(problem: dict[str, Any], model_text: str) -> dict[str, Any]:
    code = salvage_code(model_text)
    if not code:
        return {
            "ok": False,
            "n_pass": 0,
            "n_total": 1,
            "failures": ["empty_code"],
            "error": "empty_code",
            "code": "",
            "how": "none",
        }
    program, how = build_program(code, problem)
    namespace: dict[str, Any] = {"__name__": "__main__"}

    def _run() -> None:
        exec(compile(program, f"<{problem['task_id']}>", "exec"), namespace, namespace)

    try:
        run_with_exec_timeout(_run)
        return {
            "ok": True,
            "n_pass": 1,
            "n_total": 1,
            "failures": [],
            "error": None,
            "code": code,
            "how": how,
        }
    except ExecTimeout:
        return {
            "ok": False,
            "n_pass": 0,
            "n_total": 1,
            "failures": ["exec_timeout"],
            "error": "exec_timeout",
            "code": code,
            "how": how,
        }
    except Exception as exc:
        return {
            "ok": False,
            "n_pass": 0,
            "n_total": 1,
            "failures": [f"{type(exc).__name__}:{exc}"],
            "error": f"{type(exc).__name__}:{exc}",
            "code": code,
            "how": how,
        }


def load_quix_cases(name: str) -> list[tuple[list[Any], Any]]:
    raw = (DATA_DIR / f"quixbugs_{name}_cases.json").read_text(encoding="utf-8")
    cases: list[tuple[list[Any], Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        fixed = line.replace("true", "True").replace("false", "False")
        pair = ast.literal_eval(fixed)
        cases.append((list(pair[0]), pair[1]))
    return cases


def evaluate_audit(name: str, entry_point: str, model_text: str) -> dict[str, Any]:
    code = salvage_code(model_text)
    cases = load_quix_cases(name)
    namespace: dict[str, Any] = {}
    try:
        run_with_exec_timeout(lambda: exec(compile(code, f"<{name}>", "exec"), namespace, namespace))
    except Exception as exc:
        return {
            "ok": False,
            "n_pass": 0,
            "n_total": len(cases),
            "failures": [f"exec:{type(exc).__name__}:{exc}"],
            "error": f"exec:{type(exc).__name__}:{exc}",
            "code": code,
        }
    if entry_point not in namespace:
        return {
            "ok": False,
            "n_pass": 0,
            "n_total": len(cases),
            "failures": [f"missing:{entry_point}"],
            "error": f"missing:{entry_point}",
            "code": code,
        }
    fn = namespace[entry_point]
    passed = 0
    failures: list[str] = []
    for args, expected in cases:
        try:

            def _call(a=args, e=expected) -> None:
                got = fn(*a)
                if got != e:
                    raise AssertionError(f"{args} → {got!r}, expected {e!r}")

            run_with_exec_timeout(_call)
            passed += 1
        except Exception as exc:
            failures.append(f"{args}:{type(exc).__name__}:{exc}")
    return {
        "ok": passed == len(cases),
        "n_pass": passed,
        "n_total": len(cases),
        "failures": failures[:6],
        "error": None if passed == len(cases) else "assert_failures",
        "code": code,
    }


def harness_selftest() -> dict[str, Any]:
    he_path = DATA_DIR / "HumanEval.jsonl"
    problems = [
        json.loads(line)
        for line in he_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: list[dict[str, Any]] = []
    for problem in problems[:HUMANEVAL_N]:
        full = problem["prompt"] + problem["canonical_solution"]
        as_full = evaluate_humaneval(problem, f"```python\n{full}\n```")
        as_body = evaluate_humaneval(
            problem, f"```python\n{problem['canonical_solution']}\n```"
        )
        rows.append(
            {
                "task_id": problem["task_id"],
                "full_def_ok": as_full["ok"],
                "body_ok": as_body["ok"],
                "full_err": as_full.get("error"),
                "body_err": as_body.get("error"),
            }
        )
    audits = [
        ("bitcount", "bitcount"),
        ("gcd", "gcd"),
        ("is_valid_parenthesization", "is_valid_parenthesization"),
    ][:AUDIT_N]
    for name, entry in audits:
        correct = (DATA_DIR / f"quixbugs_{name}_correct.py").read_text(encoding="utf-8")
        buggy = (DATA_DIR / f"quixbugs_{name}_buggy.py").read_text(encoding="utf-8")
        good = evaluate_audit(name, entry, f"```python\n{correct}\n```")
        bad = evaluate_audit(name, entry, f"```python\n{buggy}\n```")
        rows.append(
            {
                "task_id": f"QuixBugs/{name}",
                "full_def_ok": good["ok"],
                "body_ok": not bad["ok"],
                "full_err": good.get("error"),
                "body_err": "buggy_must_fail" if bad["ok"] else None,
            }
        )
    return {"ok": all(r["full_def_ok"] and r["body_ok"] for r in rows), "rows": rows}


def load_tasks() -> list[Task]:
    he_path = DATA_DIR / "HumanEval.jsonl"
    problems = [
        json.loads(line)
        for line in he_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks: list[Task] = []
    audits = [
        ("bitcount", "bitcount"),
        ("gcd", "gcd"),
        ("is_valid_parenthesization", "is_valid_parenthesization"),
    ][:AUDIT_N]
    for name, entry in audits:
        buggy = (DATA_DIR / f"quixbugs_{name}_buggy.py").read_text(encoding="utf-8")
        buggy_fn = salvage_code(buggy) or buggy.split('"""')[0].strip()
        user = (
            "You are doing a CODE AUDIT / bug fix.\n"
            "The function below has ONE bug. Fix it.\n"
            "Output ONLY a ```python``` block with the corrected COMPLETE function. No prose.\n\n"
            f"```python\n{buggy_fn}\n```\n"
        )
        tasks.append(
            Task(
                kind="audit",
                task_id=f"QuixBugs/{name}",
                prompt_user=user,
                entry_point=entry,
                evaluate=lambda text, n=name, e=entry: evaluate_audit(n, e, text),
            )
        )
    for problem in problems[:HUMANEVAL_N]:
        user = (
            "Complete the following Python function. Output ONLY a ```python``` block "
            "with the COMPLETE function (signature + body). No prose.\n\n"
            f"```python\n{problem['prompt']}```\n"
        )
        tasks.append(
            Task(
                kind="humaneval",
                task_id=problem["task_id"],
                prompt_user=user,
                entry_point=problem["entry_point"],
                evaluate=lambda text, p=problem: evaluate_humaneval(p, text),
                he_prompt=problem["prompt"],
            )
        )
    return tasks


def repair_prompt(task: Task, code: str, score: dict[str, Any]) -> str:
    fails = list(score.get("failures") or [])
    if score.get("error"):
        fails.append(str(score["error"]))
    fail_txt = "\n".join(f"- {f}" for f in fails[:8]) or "- unknown"
    return (
        "Your previous code FAILED the official unit tests. Write a corrected version.\n"
        "Output ONLY a ```python``` block with the COMPLETE function. No prose.\n\n"
        f"Task: {task.task_id} ({task.kind})\n"
        f"Entry point: {task.entry_point}\n\n"
        f"Previous attempt:\n```python\n{code}\n```\n\n"
        f"Failures from execution:\n{fail_txt}\n\n"
        "Original task reminder:\n"
        f"{task.prompt_user}"
    )


def predictor_hit_frac(runtime: SparseDeepseekRuntime) -> float:
    hits = 0.0
    asked = 0.0
    for pred in runtime.predictors.values():
        st = pred.learning_stats()
        hits += st["predict_hit_frac"] * max(1.0, st["n_deposits"])
        asked += max(1.0, st["n_deposits"])
    return hits / max(1.0, asked)


def relieve_memory_pressure(runtime: SparseDeepseekRuntime) -> int:
    """Эмерджентный sleep + структурный потолок hot ≤ pack_dev_cap."""
    dropped_n = 0
    store = runtime.experts
    if store is not None:
        # evict_below_mean → drop_expert already takes store._lock (non-reentrant).
        dropped_n += len(store.evict_below_mean())
        if hasattr(runtime, "trim_hot_to_orbit_cap"):
            dropped_n += int(runtime.trim_hot_to_orbit_cap())  # type: ignore[attr-defined]
        else:
            runtime._drop_pack_dev_missing()
    if hasattr(runtime, "clear_fp32_cache"):
        runtime.clear_fp32_cache()  # type: ignore[attr-defined]
    gc.collect()
    return dropped_n


def reset_between_attempts(runtime: SparseDeepseekRuntime) -> None:
    runtime.clear_expert_residency()
    gc.collect()


def generate_once(
    runtime: SparseDeepseekRuntime,
    tokenizer: Any,
    prompt: str,
    *,
    prefetch: bool,
    process: psutil.Process,
) -> tuple[str, dict[str, Any]]:
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = f"User: {prompt}\n\nAssistant:"
    input_ids = tokenizer(rendered, return_tensors="pt")["input_ids"]
    runtime.use_modeled_prefetch = prefetch
    runtime.token_horizon = 4 if prefetch else 1
    runtime.prefetch_horizon = 2 if prefetch else 0
    # при prefetch держим hot, но при давлении RAM всё равно sleep
    runtime.keep_hot_during_generate = prefetch
    eos = getattr(tokenizer, "eos_token_id", None)
    pressure_line = int(runtime.spine_bytes)
    guard = {
        "peak_rss_gb": 0.0,
        "sleep_evictions": 0,
        "swap0": int(psutil.swap_memory().used),
    }

    def progress(event: dict[str, Any]) -> None:
        if event.get("phase") != "token_done":
            return
        step = int(event.get("step") or 0)
        guard["peak_rss_gb"] = max(guard["peak_rss_gb"], process.memory_info().rss / 1e9)
        mem = psutil.virtual_memory()
        swap_used = int(psutil.swap_memory().used)
        # на lean всегда подрезаем hot (иначе prefetch раздувает десятки GB)
        dropped = relieve_memory_pressure(runtime)
        guard["sleep_evictions"] += dropped
        if dropped and (mem.available < pressure_line or swap_used > guard["swap0"]):
            print(
                f"    sleep: free={mem.available / 1e9:.1f}GB swapΔ="
                f"{(swap_used - guard['swap0']) / 1e6:.0f}MB → выгрузили {dropped}",
                flush=True,
            )
        if step == 1 or step % HEARTBEAT_EVERY == 0:
            print(
                f"    пульс: tok={step}/{MAX_NEW} elapsed={event.get('elapsed_s')}s "
                f"miss_wait={event.get('gate_miss_s')}s rss={guard['peak_rss_gb']:.1f}GB "
                f"learn_hit={predictor_hit_frac(runtime):.3f}",
                flush=True,
            )

    previous_cb = runtime.progress_cb
    runtime.progress_cb = progress
    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(GEN_TIMEOUT_S)
    started = time.perf_counter()
    try:
        output = runtime.generate(
            input_ids, max_new_tokens=MAX_NEW, temperature=TEMPERATURE, eos_token_id=eos
        )
        text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True).strip()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        runtime.progress_cb = previous_cb
    wall = time.perf_counter() - started
    stats = runtime.stats()
    return text, {
        "wall_s": wall,
        "tokens_out": int(output.shape[-1]),
        "miss_wait_s": float(stats.get("gate_miss_seconds") or 0.0),
        "gate_hit": int(stats.get("n_gate_pack_hit") or 0),
        "gate_miss": int(stats.get("n_gate_pack_miss") or 0),
        "modeled_hit": int(stats.get("n_modeled_hit") or 0),
        "peak_rss_gb": guard["peak_rss_gb"],
        "sleep_evictions": guard["sleep_evictions"],
        "learn_hit_frac": predictor_hit_frac(runtime),
    }


def run_agent(
    runtime: SparseDeepseekRuntime,
    tokenizer: Any,
    task: Task,
    *,
    prefetch: bool,
    process: psutil.Process,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    prompt = task.prompt_user
    final_score: dict[str, Any] = {}
    error: str | None = None
    learn_before = predictor_hit_frac(runtime)

    for attempt_index in range(MAX_REPAIRS + 1):
        label = "candidate" if attempt_index == 0 else "repair"
        reset_between_attempts(runtime)
        print(
            f"  {label} {attempt_index + 1}/{MAX_REPAIRS + 1} "
            f"(rss={process.memory_info().rss / 1e9:.1f}GB "
            f"free={psutil.virtual_memory().available / 1e9:.1f}GB "
            f"learn_hit={learn_before:.3f})",
            flush=True,
        )
        try:
            raw, metrics = generate_once(
                runtime, tokenizer, prompt, prefetch=prefetch, process=process
            )
        except GenerationTimeout:
            error = f"GenerationTimeout>{GEN_TIMEOUT_S}s:{label}"
            print(f"  ТАЙМ-АУТ: {error}", flush=True)
            break
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            print(f"  ОШИБКА генерации: {error}", flush=True)
            traceback.print_exc()
            break
        score = task.evaluate(raw)
        attempts.append({"kind": label, "raw": raw[:2000], "score": score, **metrics})
        final_score = score
        print(
            f"  {label}: pass={score['ok']} checks={score['n_pass']}/{score['n_total']} "
            f"wall={metrics['wall_s']:.1f}s miss_wait={metrics['miss_wait_s']:.1f}s "
            f"sleep={metrics['sleep_evictions']} "
            f"code={(score.get('code') or '')[:100]!r}",
            flush=True,
        )
        if score["ok"]:
            break
        if attempt_index < MAX_REPAIRS:
            print(f"  tool feedback → repair: {score.get('failures')}", flush=True)
            prompt = repair_prompt(task, score.get("code") or salvage_code(raw), score)

    if not final_score:
        final_score = {
            "ok": False,
            "n_pass": 0,
            "n_total": 1,
            "failures": [error or "no_attempt"],
            "error": error or "no_attempt",
            "code": "",
        }
    codes = [(a.get("score") or {}).get("code") for a in attempts]
    broken = bool(error) and not any(codes)
    if attempts and not any(codes):
        broken = True
    return {
        "task_id": task.task_id,
        "kind": task.kind,
        "prefetch": prefetch,
        "passed": bool(final_score.get("ok")),
        "score": final_score,
        "attempts": attempts,
        "error": error,
        "broken": broken,
        "miss_wait_s": sum(float(a.get("miss_wait_s") or 0) for a in attempts),
        "wall_s": sum(float(a.get("wall_s") or 0) for a in attempts),
        "learn_hit_before": learn_before,
        "learn_hit_after": predictor_hit_frac(runtime),
    }


def ours_must_stop(result: dict[str, Any]) -> bool:
    if result.get("broken"):
        return True
    if result.get("error") and not result.get("attempts"):
        return True
    codes = [(a.get("score") or {}).get("code") for a in result.get("attempts") or []]
    if result.get("attempts") and not any(codes):
        return True
    return False


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = (LOGS_DIR / f"{BENCHMARK}_{stamp}.log").open("w", encoding="utf-8")
    real_stdout = sys.stdout
    sys.stdout = Tee(real_stdout, log_file)
    report_path = REPORTS_DIR / f"{BENCHMARK}_{stamp}_report.md"
    results_path = REPORTS_DIR / f"{BENCHMARK}_{stamp}_results.json"

    process = psutil.Process()
    print(f"=== {BENCHMARK} ===", flush=True)
    print(
        f"DeepSeek-V2-Lite-Chat; HumanEval×{HUMANEVAL_N} + QuixBugs×{AUDIT_N}; "
        f"MAX_NEW={MAX_NEW}; lean fp16-spine + fp32-LRU; OrbitPredictor без magic",
        flush=True,
    )
    wait_idle()

    print("СТАТУС: самопроверка стенда…", flush=True)
    selftest = harness_selftest()
    for row in selftest["rows"]:
        print(
            f"  {row['task_id']}: эталон={row['full_def_ok']} "
            f"тело/баг={row['body_ok']} {row['full_err'] or ''} {row['body_err'] or ''}",
            flush=True,
        )
    if not selftest["ok"]:
        print("СТОП: стенд кривой. Модель НЕ гоняем.")
        print("СТАТУС: VERDICT=FAIL_HARNESS_BROKEN")
        log_file.close()
        sys.stdout = real_stdout
        raise SystemExit(2)
    print("  линейка прямая", flush=True)

    tasks = load_tasks()
    print("задачи: " + ", ".join(t.task_id for t in tasks), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
    print("СТАТУС: грузим DeepSeek lean spine (fp16 + fp32-LRU)…", flush=True)
    runtime = LeanDeepseekRuntime.load_lean(
        model_id=MID,
        use_modeled_prefetch=True,
        prefetch_horizon=2,
        token_horizon=4,
    )
    print(
        f"СТАТУС: spine={runtime.spine_bytes / 1e9:.2f}GB; "
        f"RSS={process.memory_info().rss / 1e9:.1f}GB; "
        f"free={psutil.virtual_memory().available / 1e9:.1f}GB",
        flush=True,
    )
    # если после загрузки свободно меньше spine — сразу отказать, не ждать OOM
    if psutil.virtual_memory().available < runtime.spine_bytes // 2:
        print(
            "СТОП: после загрузки слишком мало свободной RAM. "
            "Классику НЕ гоняем. VERDICT=FAIL_RAM_TOO_TIGHT",
            flush=True,
        )
        log_file.close()
        sys.stdout = real_stdout
        raise SystemExit(3)

    pairs: list[dict[str, Any]] = []
    stop = False
    learn_series: list[float] = []
    for index, task in enumerate(tasks, 1):
        print(f"\n=== задача {index}/{len(tasks)}: {task.task_id} ({task.kind}) ===", flush=True)
        print("  НАША ветка (orbit prefetch + динамическое обучение)", flush=True)
        ours = run_agent(runtime, tokenizer, task, prefetch=True, process=process)
        learn_series.append(ours["learn_hit_after"])
        print(
            f"  → ours passed={ours['passed']} miss_wait={ours['miss_wait_s']:.1f}s "
            f"wall={ours['wall_s']:.1f}s learn_hit={ours['learn_hit_after']:.3f}",
            flush=True,
        )
        if ours_must_stop(ours):
            print("СТОП: наша ветка сломала пайплайн. Классику НЕ гоняем.", flush=True)
            pairs.append({"task_id": task.task_id, "kind": task.kind, "ours": ours, "classic": None})
            stop = True
            break

        print("  КЛАССИКА (без prefetch; deposit всё равно копит след)", flush=True)
        classic = run_agent(runtime, tokenizer, task, prefetch=False, process=process)
        learn_series.append(classic["learn_hit_after"])
        print(
            f"  → classic passed={classic['passed']} miss_wait={classic['miss_wait_s']:.1f}s "
            f"wall={classic['wall_s']:.1f}s learn_hit={classic['learn_hit_after']:.3f}",
            flush=True,
        )
        pairs.append(
            {"task_id": task.task_id, "kind": task.kind, "ours": ours, "classic": classic}
        )

    complete = [p for p in pairs if p["classic"] is not None]
    comparisons: dict[str, Any] = {}
    learning: dict[str, Any] = {}
    if complete:
        ours_pass = [1.0 if p["ours"]["passed"] else 0.0 for p in complete]
        classic_pass = [1.0 if p["classic"]["passed"] else 0.0 for p in complete]
        ours_miss = [p["ours"]["miss_wait_s"] for p in complete]
        classic_miss = [p["classic"]["miss_wait_s"] for p in complete]
        better_code, code_votes = emerges_greater(ours_pass, classic_pass)
        less_miss, miss_votes = emerges_greater(
            [-v for v in ours_miss], [-v for v in classic_miss]
        )
        comparisons = {
            "better_pass": {"pass": better_code, **code_votes},
            "less_miss_wait": {"pass": less_miss, **miss_votes},
            "pass_at_1_ours": sum(ours_pass) / len(ours_pass),
            "pass_at_1_classic": sum(classic_pass) / len(classic_pass),
            "mean_miss_ours": sum(ours_miss) / len(ours_miss),
            "mean_miss_classic": sum(classic_miss) / len(classic_miss),
            "n_ours_pass": int(sum(ours_pass)),
            "n_classic_pass": int(sum(classic_pass)),
            "n_tasks": len(complete),
        }
        # кривая обучения: вторая половина hit_frac vs первая (эмерджентно)
        mid = max(1, len(learn_series) // 2)
        early = learn_series[:mid]
        late = learn_series[mid:] or learn_series[-1:]
        # выравниваем длины голосованием по мин. длине
        n = min(len(early), len(late))
        learns_up, learn_votes = emerges_greater(late[:n], early[:n])
        learning = {
            "improves": learns_up,
            **learn_votes,
            "early_mean": sum(early) / len(early),
            "late_mean": sum(late) / len(late),
            "series": learn_series,
        }

    he_complete = [p for p in complete if p["kind"] == "humaneval"]
    audit_complete = [p for p in complete if p["kind"] == "audit"]

    if stop:
        verdict = "FAIL_OURS_BROKEN_STOP"
    elif not complete:
        verdict = "FAIL_NO_PAIRS"
    elif (
        comparisons["pass_at_1_ours"] == comparisons["pass_at_1_classic"]
        and comparisons["less_miss_wait"]["pass"]
    ):
        verdict = "PASS_CODE_TIE_PREFETCH_EDGE"
    elif comparisons["better_pass"]["pass"] and comparisons["pass_at_1_ours"] > 0:
        verdict = "PASS_CODE_QUALITY"
    elif comparisons["less_miss_wait"]["pass"] and learning.get("improves"):
        verdict = "PASS_PREFETCH_AND_LEARNING"
    elif comparisons["less_miss_wait"]["pass"]:
        verdict = "PASS_PREFETCH_PARTIAL"
    elif comparisons["pass_at_1_ours"] < comparisons["pass_at_1_classic"]:
        verdict = "FAIL_LOSES_TO_CLASSIC_CODE"
    elif comparisons["pass_at_1_ours"] == 0:
        verdict = "FAIL_BOTH_ZERO_CODE"
    else:
        verdict = "FAIL_NO_PREFETCH_GAIN"

    lines = [
        f"# {BENCHMARK}",
        "",
        f"**verdict:** {verdict}",
        "",
        "## Простыми словами",
        "",
        "DeepSeek-V2-Lite на ноутбуке: орбита учится без magic numbers",
        "(пороги из локальной статистики окна/поля), эксперты подгружаются по предсказанию,",
        "при нехватке RAM — sleep. Сначала наша ветка, потом классика.",
        "",
        f"- pass@1: ours={comparisons.get('pass_at_1_ours', 0):.2f} "
        f"({comparisons.get('n_ours_pass', 0)}/{comparisons.get('n_tasks', 0)}), "
        f"classic={comparisons.get('pass_at_1_classic', 0):.2f}",
        f"- HumanEval ours: "
        f"{sum(1 for p in he_complete if p['ours']['passed'])}/{len(he_complete) or 0}",
        f"- QuixBugs audit ours: "
        f"{sum(1 for p in audit_complete if p['ours']['passed'])}/{len(audit_complete) or 0}",
    ]
    if comparisons:
        lines += [
            f"- Меньше ожидания экспертов: **{comparisons['less_miss_wait']['pass']}** "
            f"(wins={comparisons['less_miss_wait']['wins']} "
            f"losses={comparisons['less_miss_wait']['losses']})",
            f"- Среднее ожидание: ours={comparisons['mean_miss_ours']:.1f}s, "
            f"classic={comparisons['mean_miss_classic']:.1f}s",
        ]
    if learning:
        lines += [
            f"- Обучение орбиты растёт к концу: **{learning['improves']}** "
            f"(wins={learning['wins']} losses={learning['losses']}; "
            f"early={learning['early_mean']:.3f} late={learning['late_mean']:.3f})",
        ]
    lines += [
        "",
        "## По задачам",
        "",
        "| task | kind | ours | classic | ours miss | classic miss | learn_hit |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for p in pairs:
        c = p["classic"]
        if c is None:
            lines.append(
                f"| {p['task_id']} | {p['kind']} | {int(p['ours']['passed'])} | — | "
                f"{p['ours']['miss_wait_s']:.1f} | — | {p['ours']['learn_hit_after']:.3f} |"
            )
        else:
            lines.append(
                f"| {p['task_id']} | {p['kind']} | {int(p['ours']['passed'])} | "
                f"{int(c['passed'])} | {p['ours']['miss_wait_s']:.1f} | "
                f"{c['miss_wait_s']:.1f} | {p['ours']['learn_hit_after']:.3f} |"
            )
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    results_path.write_text(
        json.dumps(
            {
                "verdict": verdict,
                "stop": stop,
                "model_id": MID,
                "comparisons": comparisons,
                "learning": learning,
                "pairs": pairs,
                "selftest": selftest,
                "config": {
                    "max_new": MAX_NEW,
                    "max_repairs": MAX_REPAIRS,
                    "humaneval_n": HUMANEVAL_N,
                    "audit_n": AUDIT_N,
                    "dtype": "float32",
                    "orbit_predictor": "emergent_no_magic",
                },
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nСТАТУС: VERDICT={verdict}", flush=True)
    print(f"report={report_path}", flush=True)
    log_file.close()
    sys.stdout = real_stdout
    if verdict.startswith("FAIL"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
