# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.5.x | Yes (current) |
| < 0.5 | Best-effort only |

This is a **research prototype**, not a hardened production inference server.

## Reporting a vulnerability

Please open a **private** GitHub security advisory on this repository, or contact the maintainer via GitHub profile if advisories are unavailable.

Include:

1. Description and impact (e.g. unexpected weight load, path traversal in shard resolve).
2. Minimal reproduction steps.
3. Affected commit / tag.

Do **not** open a public issue for exploitable local-file or supply-chain problems until a fix is available.

## Non-security issues

Bugs, OOM on CUDA, and audit findings → public Issues, referencing [docs/AUDITOR_ISSUES.md](docs/AUDITOR_ISSUES.md) when relevant.

## Supply chain notes

- Model weights are downloaded from Hugging Face under DeepSeek/HF terms — not from this git tree.
- Prefer pinned dependency ranges in your environment when embedding this package.
