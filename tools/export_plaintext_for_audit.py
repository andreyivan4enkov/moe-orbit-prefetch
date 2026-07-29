#!/usr/bin/env python3
"""
Export a plain-text bundle of all .py / .md / .json (text) sources for LLM audit.

Gemini / other tools sometimes ingest a ZIP as UTF-8 and see Deflate garbage
(`PK`, high-entropy bytes). This script writes ONE readable .txt file instead.

Usage:
  python tools/export_plaintext_for_audit.py
  python tools/export_plaintext_for_audit.py --out /tmp/moe_orbit_audit.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "egg-info"}
TEXT_SUFFIX = {".py", ".md", ".txt", ".toml", ".cff", ".json", ".yml", ".yaml", ".cfg", ".ini"}
# Never dump binaries into the audit text
SKIP_SUFFIX = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".gz", ".safetensors", ".bin", ".pt", ".pth"}


def iter_text_files(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in p.parts):
            continue
        if p.suffix.lower() in SKIP_SUFFIX:
            continue
        if p.suffix.lower() not in TEXT_SUFFIX and p.name not in {"LICENSE", "NOTICE", "LICENSE.txt"}:
            continue
        yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "exports" / "PLAINTEXT_AUDIT_BUNDLE.txt",
    )
    args = ap.parse_args()
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    parts.append("# moe-orbit-prefetch PLAINTEXT AUDIT BUNDLE\n")
    parts.append("# Generated for LLM review. Binaries (PNG etc.) omitted on purpose.\n")
    parts.append(f"# Root: {ROOT}\n\n")

    n = 0
    for p in iter_text_files(ROOT):
        rel = p.relative_to(ROOT).as_posix()
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            parts.append(f"\n\n##### SKIP binary/non-utf8: {rel}\n")
            continue
        n += 1
        parts.append(f"\n\n{'=' * 72}\n")
        parts.append(f"FILE: {rel}\n")
        parts.append(f"{'=' * 72}\n\n")
        parts.append(text)
        if not text.endswith("\n"):
            parts.append("\n")

    out.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {out} files={n} bytes={out.stat().st_size}")
    print("Feed THIS .txt to Gemini (or paste sections). Do NOT feed a .zip as text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
