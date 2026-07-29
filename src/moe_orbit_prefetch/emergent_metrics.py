"""
Эмерджентные гейты: сравнения без магических порогов PASS.

PASS = большинство локальных сравнений (каждый шаг голосует).
Пороги отбора в динамике — из локальной статистики поля/окна.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def numerical_eps() -> float:
    return float(np.finfo(np.float64).eps)


def emerges_greater(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
) -> tuple[bool, dict[str, Any]]:
    """
    Ряд a эмерджентно выше ряда b по локальным голосам:
      wins = #{i: a_i > b_i}, losses = #{i: a_i < b_i}
      PASS iff wins > losses
    """
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    n = int(min(aa.size, bb.size))
    eps = numerical_eps()
    if n <= 0:
        return False, {"gap": 0.0, "mad": eps, "n": 0, "wins": 0, "losses": 0, "win_frac": 0.0}
    d = aa[:n] - bb[:n]
    wins = int(np.sum(d > 0))
    losses = int(np.sum(d < 0))
    gap = float(np.mean(d))
    mad = float(np.mean(np.abs(d - gap))) + eps
    return bool(wins > losses), {
        "gap": gap,
        "mad": mad,
        "n": n,
        "wins": wins,
        "losses": losses,
        "win_frac": wins / max(wins + losses, 1),
    }


def emerges_positive(x: np.ndarray | list[float]) -> tuple[bool, dict[str, Any]]:
    """Сигнал эмерджентно > 0: большинство локальных x_i > 0."""
    xx = np.asarray(x, dtype=np.float64).ravel()
    return emerges_greater(xx, np.zeros_like(xx))


def local_mean_threshold(values: np.ndarray | list[float], *, positive_only: bool = True) -> float:
    """Порог отбора = среднее локального поля."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if positive_only:
        v = v[v > 0]
    if v.size == 0:
        return 0.0
    return float(np.mean(v))


def local_match_floor(cosines: np.ndarray | list[float]) -> float:
    """Пол fuzzy-match = max(0, mean(cos)) по окну кандидатов."""
    c = np.asarray(cosines, dtype=np.float64).ravel()
    if c.size == 0:
        return 0.0
    return max(0.0, float(np.mean(c)))


def deposit_scale(s_env_masses: np.ndarray | list[float], resonance: float) -> float:
    """
    Депозит из локального резонанса и уже накопленного поля:
      scale = (1 - R) * (1 + mean(|S|))
    Пустое поле → (1-R)*1 — след всё равно появляется из события.
    """
    m = np.asarray(s_env_masses, dtype=np.float64).ravel()
    mean_lvl = float(np.mean(np.abs(m))) if m.size else 0.0
    r = max(0.0, min(1.0, float(resonance)))
    return (1.0 - r) * (1.0 + mean_lvl) + numerical_eps()


def decay_hold_from_resonance(mean_resonance: float) -> float:
    """Доля сохранения следа = clip(R, eps, 1) — высокий R держит тропы."""
    r = max(0.0, min(1.0, float(mean_resonance)))
    return max(numerical_eps(), r)


def write_mix_from_delta(delta_f: float, resonance: float) -> float:
    """Доля prev-write: R / (R + |ΔF| + eps)."""
    eps = numerical_eps()
    r = max(0.0, float(resonance))
    return r / (r + abs(float(delta_f)) + eps)


def bytes_unit_from_dim(d_model: int, expert_mult: int = 1) -> int:
    """Единица байт эксперта из размерности (bf16 * d * 4d)."""
    d = max(1, int(d_model))
    return int(2 * d * (4 * d) * max(1, int(expert_mult)))
