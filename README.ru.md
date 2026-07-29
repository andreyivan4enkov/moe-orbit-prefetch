# moe-orbit-prefetch (RU)

**Объект A** — подгрузка экспертов MoE по орбите из `h` (не весь MoE в RAM).

Подходит к **открытым MoE семейства DeepSeek** — в лаборатории проверены **DeepSeek-V2-Lite** и **GigaChat** (10B/20B). Это не «любая нейросеть» и не Mixtral без нового адаптера.

| Статус | research prototype (alpha) |
| Лицензия | Apache-2.0 |
| Веса | не в git |
| Железо автора | MacBook-класс; полный бенч на 100 задач / GPU **не заявлен** |

См. [docs/SUPPORTED_MODELS.md](docs/SUPPORTED_MODELS.md), [MODEL_CARD.md](MODEL_CARD.md), [docs/LAB_SCOPE.md](docs/LAB_SCOPE.md).

## Живые результаты (Tier L)

| Файл | Честно |
|---|---|
| [results/v13_…](results/v13_dynamic_expert_slice.md) / [v36_…](results/v36_humaneval_prefetch_edge.md) | DeepSeek: residency + lean code (tie) + лучший miss-wait |
| [results/gigachat_v21_…](results/gigachat_v21_orbit_apply.md) | GigaChat-10B smoke орбиты |
| [results/gigachat_v32_…](results/gigachat_v32_lean.md) / [v34_…](results/gigachat_v34_humaneval.md) | GigaChat-20B: miss-wait лучше; код tie |

English: [README.md](README.md).
