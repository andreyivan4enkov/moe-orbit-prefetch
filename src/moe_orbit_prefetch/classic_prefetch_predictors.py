"""
Classical prefetch predictors with the same duck-typed API as OrbitPredictor.

Used for live miss-wait head-to-heads (ours first → classic).
Not a substitute for OrbitPredictor — baselines only.

License: Apache-2.0
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _BaseClassic:
    n_experts: int = 64
    top_k: int = 6
    window: int = 48
    n_deposits: int = 0
    n_predict_hits: int = 0
    n_predict_asked: int = 0

    def _cold(self) -> list[int]:
        return list(range(min(self.top_k, self.n_experts)))

    def _record_hit(self, pred: list[int], true_experts: list[int]) -> None:
        hit = len(set(pred) & set(int(e) for e in true_experts))
        self.n_predict_hits += hit
        self.n_predict_asked += max(1, len(true_experts))

    def prefetch_set(
        self, h: np.ndarray, tok_id: int | None, horizon_copies: int = 1
    ) -> list[int]:
        pred = self.predict(h, tok_id)
        copies = max(1, int(horizon_copies))
        if copies <= 1:
            return pred
        wide_k = min(self.n_experts, self.top_k * copies)
        # widen by cycling next ids after the predicted set (deterministic, no h-topology)
        out = list(pred)
        cursor = (out[-1] + 1) % self.n_experts if out else 0
        while len(out) < wide_k:
            if cursor not in out:
                out.append(cursor)
            cursor = (cursor + 1) % self.n_experts
        return out

    def predict_token_horizon(
        self,
        h: np.ndarray,
        tok_id: int | None,
        H: int,
    ) -> list[list[int]]:
        H = max(1, int(H))
        out: list[list[int]] = [self.predict(h, tok_id)]
        for step in range(1, H):
            out.append(self.prefetch_set(h, tok_id, horizon_copies=1 + min(step, 2)))
        return out

    def learning_stats(self) -> dict[str, float]:
        asked = max(1, self.n_predict_asked)
        return {
            "n_deposits": float(self.n_deposits),
            "predict_hit_frac": self.n_predict_hits / asked,
            "field_mean_pos": 0.0,
            "hist_len": 0.0,
        }

    def predict(self, h: np.ndarray, tok_id: int | None = None) -> list[int]:
        raise NotImplementedError

    def deposit(
        self, h: np.ndarray, true_experts: list[int], tok_id: int | None = None
    ) -> None:
        raise NotImplementedError


@dataclass
class FrequencyPredictor(_BaseClassic):
    """Prefetch the historically most frequent live experts."""

    counts: Counter = field(default_factory=Counter)

    def predict(self, h: np.ndarray, tok_id: int | None = None) -> list[int]:
        del h, tok_id
        if not self.counts:
            return self._cold()
        return [int(e) for e, _ in self.counts.most_common(self.top_k)]

    def deposit(
        self, h: np.ndarray, true_experts: list[int], tok_id: int | None = None
    ) -> None:
        pred = self.predict(h, tok_id)
        self._record_hit(pred, true_experts)
        for e in true_experts:
            if 0 <= int(e) < self.n_experts:
                self.counts[int(e)] += 1
        self.n_deposits += 1

    def learning_stats(self) -> dict[str, float]:
        st = super().learning_stats()
        st["hist_len"] = float(sum(self.counts.values()))
        return st


@dataclass
class LruPredictor(_BaseClassic):
    """
    Recent-hot / LRU-style: experts seen most recently score highest.
    Recency clock increments on each deposit; predict top_k by last-seen time.
    """

    last_seen: dict[int, int] = field(default_factory=dict)
    clock: int = 0

    def predict(self, h: np.ndarray, tok_id: int | None = None) -> list[int]:
        del h, tok_id
        if not self.last_seen:
            return self._cold()
        ranked = sorted(self.last_seen.items(), key=lambda kv: kv[1], reverse=True)
        return [int(e) for e, _ in ranked[: self.top_k]]

    def deposit(
        self, h: np.ndarray, true_experts: list[int], tok_id: int | None = None
    ) -> None:
        pred = self.predict(h, tok_id)
        self._record_hit(pred, true_experts)
        self.clock += 1
        for e in true_experts:
            eid = int(e)
            if 0 <= eid < self.n_experts:
                self.last_seen[eid] = self.clock
        # prune to stand window size (keep most recent experts only)
        if len(self.last_seen) > self.window * self.top_k:
            keep = {
                e: t
                for e, t in sorted(
                    self.last_seen.items(), key=lambda kv: kv[1], reverse=True
                )[: self.window * self.top_k]
            }
            self.last_seen = keep
        self.n_deposits += 1


@dataclass
class PrevCopyPredictor(_BaseClassic):
    """Copy the previous step's live experts (Markov-1 / sticky)."""

    prev: list[int] = field(default_factory=list)
    last_succ: dict[int, list[int]] = field(default_factory=dict)

    def predict(self, h: np.ndarray, tok_id: int | None = None) -> list[int]:
        del h
        if tok_id is not None and tok_id in self.last_succ:
            return list(self.last_succ[tok_id])[: self.top_k]
        if self.prev:
            return list(self.prev)[: self.top_k]
        return self._cold()

    def deposit(
        self, h: np.ndarray, true_experts: list[int], tok_id: int | None = None
    ) -> None:
        pred = self.predict(h, tok_id)
        self._record_hit(pred, true_experts)
        self.prev = [int(e) for e in true_experts if 0 <= int(e) < self.n_experts][
            : self.top_k
        ]
        if tok_id is not None:
            self.last_succ[tok_id] = list(self.prev)
            # keep last_succ bounded by stand window
            if len(self.last_succ) > self.window:
                # drop oldest insertion order approx via deque of keys
                keys = list(self.last_succ.keys())
                for k in keys[: len(keys) - self.window]:
                    self.last_succ.pop(k, None)
        self.n_deposits += 1
