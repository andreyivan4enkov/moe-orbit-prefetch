"""
process_gate — не стартовать тяжёлую работу, пока жив чужой прогон.

Использование:
  from process_gate import wait_until_idle
  wait_until_idle(reason="dynamic weight smoke")
"""
from __future__ import annotations

import os
import re
import time
from typing import Callable

# Паттерны чужих/своих тяжёлых задач (не включаем сам process_gate / IDE)
_BUSY_RE = re.compile(
    r"python.*(sparse-stigmergy|megaattractor|run_all|run_v2|hf_hub|huggingface-cli|"
    r"transformers\.|embed_induction|moe_|emergent_revalidate|mabs_|ucl_)",
    re.I,
)


def _list_busy_pids(extra_filter: Callable[[str], bool] | None = None) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    my_pid = os.getpid()
    try:
        import subprocess

        raw = subprocess.check_output(["ps", "ax", "-o", "pid=,command="], text=True)
    except Exception:
        return out
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == my_pid:
            continue
        cmd = parts[1]
        # только реальный интерпретатор python, не zsh/bash-обёртка с тем же текстом команды
        if not re.search(r"(^|[\s/])python(\d+(?:\.\d+)*)?(\s|$)", cmd):
            continue
        if not _BUSY_RE.search(cmd):
            continue
        # игнор grep/pgrep/process_gate
        if "process_gate" in cmd or "pgrep" in cmd:
            continue
        # не ждать собственного дерева: ppid chain
        if pid == my_pid or _is_ancestor(pid, my_pid) or _is_ancestor(my_pid, pid):
            continue
        if extra_filter and not extra_filter(cmd):
            continue
        out.append((pid, cmd))
    return out


def _is_ancestor(ancestor: int, descendant: int) -> bool:
    """True if ancestor is in parent chain of descendant."""
    try:
        import subprocess

        pid = descendant
        for _ in range(32):
            if pid <= 1:
                return False
            if pid == ancestor:
                return True
            raw = subprocess.check_output(["ps", "-o", "ppid=", "-p", str(pid)], text=True).strip()
            if not raw:
                return False
            pid = int(raw)
    except Exception:
        return False
    return False


def wait_until_idle(
    *,
    reason: str = "",
    poll_sec: float | None = None,
    timeout_sec: float | None = None,
    log: Callable[[str], None] | None = print,
) -> None:
    """
    Ждать, пока не останется чужих совпадающих процессов.
    poll/timeout эмерджентно: poll от числа busy (больше busy → чаще смотреть нельзя —
    наоборот реже: poll = 2 + len(busy)); timeout=None → ждать бесконечно.
    """
    t0 = time.time()
    while True:
        busy = _list_busy_pids()
        if not busy:
            if log:
                msg = "process_gate: idle"
                if reason:
                    msg += f" ({reason})"
                log(msg)
            return
        if timeout_sec is not None and (time.time() - t0) > timeout_sec:
            raise TimeoutError(
                f"process_gate timeout after {timeout_sec}s; still busy: {busy[:3]}"
            )
        # эмерджентный интервал: чем больше конкурентов, тем дольше ждём между проверками
        delay = float(poll_sec) if poll_sec is not None else (2.0 + float(len(busy)))
        if log:
            pids = ",".join(str(p) for p, _ in busy[:5])
            log(f"process_gate: waiting on pids=[{pids}] … sleep {delay:.1f}s ({reason})")
        time.sleep(delay)
