# moe-orbit-prefetch (RU)

**Объект A** — подгрузка экспертов MoE по орбите из `h` (не весь MoE в RAM).

| Статус | research prototype (alpha) |
| Лицензия | Apache-2.0 |
| Веса | не в git (DeepSeek/HF) |
| Железо автора | MacBook-класс; полный бенч на 100 задач / GPU **не заявлен** |

Подробности: [MODEL_CARD.md](MODEL_CARD.md), [docs/LAB_SCOPE.md](docs/LAB_SCOPE.md), [docs/EVIDENCE_TIERS.md](docs/EVIDENCE_TIERS.md).

## Живые результаты (Tier L)

| Файл | Честно |
|---|---|
| [results/v13_dynamic_expert_slice.md](results/v13_dynamic_expert_slice.md) | Срез эксперта + сон → меньше резидентной памяти |
| [results/v36_humaneval_prefetch_edge.md](results/v36_humaneval_prefetch_edge.md) | Код как у classic; miss-wait лучше; поздний hit **не** вырос |

## Запуск на реальной модели

```bash
pip install -e ".[runtime]"
python examples/02_smoke_expert_slice.py
python examples/04_chat_ask.py "Привет" --max-new 32
python examples/bench_humaneval_lean/bench_humaneval_lean.py
```

Нужен кэш Hugging Face с DeepSeek-V2-Lite-Chat.

## Синтетика (Tier S) — не замена live

```bash
pip install -e ".[analysis]"
python analysis/generate_orbit_trajectory.py
```

English README: [README.md](README.md).
