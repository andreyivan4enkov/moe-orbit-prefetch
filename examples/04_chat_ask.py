#!/usr/bin/env python3
"""
Example 04 — one-shot chat via full sparse DeepSeek runtime + orbit prefetch.

Requires DeepSeek-V2-Lite-Chat weights in Hugging Face cache (not shipped).
Long on CPU / 16GB laptop.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default="Say hello in one short sentence.")
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()

    from moe_orbit_prefetch.deepseek_chat_engine import ask

    print("Loading sparse+orbit DeepSeek (first call may take a while)…", flush=True)
    reply = ask(args.question, max_new_tokens=args.max_new)
    print("--- reply ---")
    print(reply)
    return 0 if reply else 2


if __name__ == "__main__":
    raise SystemExit(main())
