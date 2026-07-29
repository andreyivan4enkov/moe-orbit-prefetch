"""
Modeled orbit predictor: induction + S_env по residual/эмбеддингу h.

Не заменяет live gate — предсказывает орбиту для prefetch до/во время Exec.

Пороги и масштабы — только из локальной статистики окна/поля
(emergent_metrics), без подогнанных констант PASS/депозита/затухания.
Структурные факты: n_experts, top_k, размер окна стенда.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .emergent_metrics import (
    decay_hold_from_resonance,
    deposit_scale,
    local_match_floor,
    local_mean_threshold,
    numerical_eps,
)


@dataclass
class OrbitPredictor:
    n_experts: int = 64
    top_k: int = 6
    window: int = 48  # размер стенда: сколько последних (h, orbit) держим
    s_env: np.ndarray = field(init=False)
    last_succ: dict[int, list[int]] = field(default_factory=dict)
    hist_h: list[np.ndarray] = field(default_factory=list)
    hist_exp: list[list[int]] = field(default_factory=list)
    # диагностика обучения (не пороги)
    n_deposits: int = 0
    n_predict_hits: int = 0  # сколько экспертов live ∩ predict
    n_predict_asked: int = 0

    def __post_init__(self) -> None:
        self.s_env = np.zeros(self.n_experts, dtype=np.float64)

    def _unit(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=np.float64).ravel()
        return h / (np.linalg.norm(h) + numerical_eps())

    def _window_cosines(self, h: np.ndarray) -> np.ndarray:
        if not self.hist_h:
            return np.zeros(0, dtype=np.float64)
        start = max(0, len(self.hist_h) - self.window)
        return np.asarray(
            [float(np.dot(self.hist_h[j], h)) for j in range(start, len(self.hist_h))],
            dtype=np.float64,
        )

    def _field_scale(self) -> float:
        """Характерный масштаб поля = mean положительных масс (или ε)."""
        thr = local_mean_threshold(self.s_env, positive_only=True)
        return thr if thr > 0 else numerical_eps()

    def predict(self, h: np.ndarray, tok_id: int | None = None) -> list[int]:
        h = self._unit(h)
        scores = self.s_env.copy()
        field = self._field_scale()

        # induction: эксперты, которые шли после этого токена раньше —
        # усиление в масштабе текущего поля, не фиксированное «+2»
        if tok_id is not None and tok_id in self.last_succ:
            for e in self.last_succ[tok_id]:
                if 0 <= e < self.n_experts:
                    scores[e] += field

        cosines = self._window_cosines(h)
        match_floor = local_match_floor(cosines)  # mean(cos) окна, ≥ 0
        start = max(0, len(self.hist_h) - self.window)
        for j, cos in zip(range(start, len(self.hist_h)), cosines):
            if cos > match_floor:
                for e in self.hist_exp[j]:
                    if 0 <= e < self.n_experts:
                        scores[e] += float(cos)

        if float(scores.max()) <= 0:
            # cold: детерминированный слот от направления h (не random)
            base = int(abs(h[0]) * 1e6) % self.n_experts
            return [(base + i) % self.n_experts for i in range(self.top_k)]
        return [int(i) for i in np.argsort(-scores)[: self.top_k]]

    def deposit(self, h: np.ndarray, true_experts: list[int], tok_id: int | None = None) -> None:
        h = self._unit(h)
        # качество предсказания ДО обновления следа (честная кривая обучения)
        pred_before = self.predict(h, tok_id)
        hit = len(set(pred_before) & set(int(e) for e in true_experts))
        self.n_predict_hits += hit
        self.n_predict_asked += max(1, len(true_experts))

        cosines = self._window_cosines(h)
        resonance = local_match_floor(cosines)  # локальный резонанс с окном
        # если окна ещё нет — резонанс 0 → депозит максимальный из формулы
        scale = deposit_scale(self.s_env, resonance)
        n = max(1, len(true_experts))
        for e in true_experts:
            if 0 <= e < self.n_experts:
                self.s_env[e] += scale / n

        # высокий резонанс держит тропы; пустое окно — не стираем первые следы
        if len(self.hist_h) > 0:
            self.s_env *= decay_hold_from_resonance(resonance)

        if tok_id is not None:
            self.last_succ[tok_id] = list(true_experts)
        self.hist_h.append(h)
        self.hist_exp.append(list(true_experts))
        if len(self.hist_h) > self.window + 8:
            self.hist_h = self.hist_h[-(self.window + 8) :]
            self.hist_exp = self.hist_exp[-(self.window + 8) :]
        self.n_deposits += 1

    def prefetch_set(self, h: np.ndarray, tok_id: int | None, horizon_copies: int = 1) -> list[int]:
        """Орбита + расширение горизонта: ширина = top_k × copies (структурно)."""
        pred = self.predict(h, tok_id)
        copies = max(1, int(horizon_copies))
        if copies <= 1:
            return pred
        h = self._unit(h)
        scores = self.s_env.copy()
        field = self._field_scale()
        if tok_id is not None and tok_id in self.last_succ:
            for e in self.last_succ[tok_id]:
                if 0 <= e < self.n_experts:
                    scores[e] += field
        cosines = self._window_cosines(h)
        match_floor = local_match_floor(cosines)
        start = max(0, len(self.hist_h) - self.window)
        for j, cos in zip(range(start, len(self.hist_h)), cosines):
            if cos > match_floor:
                for e in self.hist_exp[j]:
                    if 0 <= e < self.n_experts:
                        # вес шага горизонта: доля от косинуса по номеру копии
                        scores[e] += float(cos) / copies
        wide_k = min(self.n_experts, self.top_k * copies)
        if float(scores.max()) <= 0:
            return pred
        return [int(i) for i in np.argsort(-scores)[:wide_k]]

    def predict_token_horizon(
        self,
        h: np.ndarray,
        tok_id: int | None,
        H: int,
    ) -> list[list[int]]:
        """
        Орбиты на t…t+H-1: h → будущие experts до Exec.
        t: полный predict; дальше — шире поле + successor prior.
        """
        H = max(1, int(H))
        out: list[list[int]] = [self.predict(h, tok_id)]
        succ = list(self.last_succ.get(tok_id, [])) if tok_id is not None else []
        for step in range(1, H):
            wide = self.prefetch_set(h, None, horizon_copies=1 + min(step, 2))
            if succ:
                merged = list(dict.fromkeys(succ[: self.top_k] + wide))[
                    : max(self.top_k, len(wide))
                ]
                out.append(merged)
            else:
                out.append(wide)
        return out

    def learning_stats(self) -> dict[str, float]:
        asked = max(1, self.n_predict_asked)
        return {
            "n_deposits": float(self.n_deposits),
            "predict_hit_frac": self.n_predict_hits / asked,
            "field_mean_pos": local_mean_threshold(self.s_env, positive_only=True),
            "hist_len": float(len(self.hist_h)),
        }
