"""
DynamicExpertStore — подгрузка тензоров экспертов MoE по орбите (не полный model).

L0: index.json
L2: get_expert(layer, expert_id) → dict[str, Tensor]; evict по S_env / явный drop.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


_EXPERT_RE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<eid>\d+)\.(?P<param>.+)"
)
_SHARED_RE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.shared_experts\.(?P<param>.+)"
)
_GATE_RE = re.compile(r"model\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight")


@dataclass
class TensorRef:
    name: str
    shard: str


@dataclass
class DynamicExpertStore:
    model_id: str
    index_path: Path
    weight_map: dict[str, str]
    shards_dir: Path | None = None  # if shards are beside index; else resolve via hub cache
    hot: dict[tuple[int, int], dict[str, torch.Tensor]] = field(default_factory=dict)
    s_env: dict[tuple[int, int], float] = field(default_factory=dict)
    bytes_loaded: int = 0
    bytes_evicted: int = 0
    n_loads: int = 0
    n_hits: int = 0
    n_misses: int = 0
    allow_hub_download: bool = False
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_index_file(
        cls,
        model_id: str,
        index_path: Path | str,
        *,
        allow_hub_download: bool = False,
    ) -> "DynamicExpertStore":
        p = Path(index_path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            model_id=model_id,
            index_path=p,
            weight_map=dict(data["weight_map"]),
            allow_hub_download=allow_hub_download,
        )

    def list_expert_ids(self, layer: int) -> list[int]:
        ids: set[int] = set()
        prefix = f"model.layers.{layer}.mlp.experts."
        for name in self.weight_map:
            if name.startswith(prefix):
                m = _EXPERT_RE.match(name)
                if m:
                    ids.add(int(m.group("eid")))
        return sorted(ids)

    def expert_tensor_names(self, layer: int, expert_id: int) -> list[str]:
        prefix = f"model.layers.{layer}.mlp.experts.{expert_id}."
        return sorted(n for n in self.weight_map if n.startswith(prefix))

    def shard_for_expert(self, layer: int, expert_id: int) -> str | None:
        names = self.expert_tensor_names(layer, expert_id)
        if not names:
            return None
        shards = {self.weight_map[n] for n in names}
        if len(shards) != 1:
            # rare split — take first deterministically
            return sorted(shards)[0]
        return next(iter(shards))

    def resolve_shard_path(self, shard_name: str) -> Path | None:
        """Local-only first; optional hub download if allow_hub_download."""
        from huggingface_hub import try_to_load_from_cache, hf_hub_download

        cached = try_to_load_from_cache(self.model_id, shard_name)
        if cached and Path(cached).exists() and Path(cached).stat().st_size > 1_000_000:
            return Path(cached)
        # incomplete / missing
        if not self.allow_hub_download:
            return None
        path = hf_hub_download(self.model_id, shard_name)
        p = Path(path)
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
        return None

    def local_shard_status(self, shard_name: str) -> dict[str, Any]:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(self.model_id, shard_name)
        if not cached:
            return {"shard": shard_name, "present": False, "path": None, "bytes": 0}
        p = Path(cached)
        if not p.exists():
            return {"shard": shard_name, "present": False, "path": str(p), "bytes": 0}
        sz = p.stat().st_size
        return {
            "shard": shard_name,
            "present": sz > 1_000_000,
            "path": str(p),
            "bytes": sz,
            "incomplete": sz <= 1_000_000,
        }

    def get_expert(
        self,
        layer: int,
        expert_id: int,
        *,
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        key = (layer, expert_id)
        with self._lock:
            if key in self.hot:
                self.n_hits += 1
                self.s_env[key] = self.s_env.get(key, 0.0) + 1.0
                return self.hot[key]

        self.n_misses += 1
        names = self.expert_tensor_names(layer, expert_id)
        if not names:
            raise KeyError(f"no tensors for layer={layer} expert={expert_id}")

        loaded: dict[str, torch.Tensor] = {}
        nbytes = 0
        # каждый тензор может быть в своём шарде
        by_shard: dict[str, list[str]] = {}
        for name in names:
            by_shard.setdefault(self.weight_map[name], []).append(name)
        for shard_name, tnames in by_shard.items():
            path = self.resolve_shard_path(shard_name)
            if path is None:
                raise FileNotFoundError(
                    f"shard {shard_name} not local; set allow_hub_download=True to fetch"
                )
            with safe_open(str(path), framework="pt", device=device) as f:
                available = set(f.keys())
                for name in tnames:
                    if name not in available:
                        raise KeyError(f"missing tensor {name} in {shard_name}")
                    t = f.get_tensor(name)
                    short = name.split(f"experts.{expert_id}.")[-1]
                    loaded[short] = t
                    nbytes += int(t.numel() * t.element_size())

        with self._lock:
            if key in self.hot:
                self.n_hits += 1
                self.s_env[key] = self.s_env.get(key, 0.0) + 1.0
                return self.hot[key]
            self.hot[key] = loaded
            self.bytes_loaded += nbytes
            self.n_loads += 1
            self.s_env[key] = self.s_env.get(key, 0.0) + 1.0
            return loaded

    def resident_bytes(self) -> int:
        total = 0
        for pack in self.hot.values():
            seen: set[int] = set()
            for t in pack.values():
                if id(t) in seen:
                    continue
                seen.add(id(t))
                total += int(t.numel() * t.element_size())
        return total

    def evict_below_mean(self) -> list[tuple[int, int]]:
        """Sleep: выгрузить экспертов с S_env < mean(positive). Эмерджентный порог."""
        if not self.s_env:
            return []
        pos = [v for v in self.s_env.values() if v > 0]
        if not pos:
            return []
        thr = sum(pos) / len(pos)
        dropped: list[tuple[int, int]] = []
        for key, mass in list(self.s_env.items()):
            if mass < thr and key in self.hot:
                pack = self.hot.pop(key)
                seen: set[int] = set()
                for t in pack.values():
                    if id(t) in seen:
                        continue
                    seen.add(id(t))
                    self.bytes_evicted += int(t.numel() * t.element_size())
                dropped.append(key)
        return dropped

    def trim_to_bytes(self, max_bytes: int) -> list[tuple[int, int]]:
        """Жёсткий sleep: выгружать самых холодных, пока resident > max_bytes."""
        dropped: list[tuple[int, int]] = []
        while self.resident_bytes() > max_bytes and self.hot:
            # самый низкий S_env среди hot
            key = min(self.hot.keys(), key=lambda k: self.s_env.get(k, 0.0))
            pack = self.hot.pop(key)
            seen: set[int] = set()
            for t in pack.values():
                if id(t) in seen:
                    continue
                seen.add(id(t))
                self.bytes_evicted += int(t.numel() * t.element_size())
            dropped.append(key)
        return dropped

    def n_hot(self) -> int:
        return len(self.hot)

    def stats(self) -> dict[str, Any]:
        return {
            "n_hot": self.n_hot(),
            "resident_bytes": self.resident_bytes(),
            "bytes_loaded": self.bytes_loaded,
            "bytes_evicted": self.bytes_evicted,
            "n_loads": self.n_loads,
            "n_hits": self.n_hits,
            "n_misses": self.n_misses,
            "s_env_keys": len(self.s_env),
        }
