# moe-orbit-prefetch (RU)

**Объект A** — подгрузка экспертов MoE по орбите из `h` (не весь MoE в RAM).

Подходит к **открытым MoE семейства DeepSeek** — в лаборатории проверены **DeepSeek-V2-Lite** и **GigaChat** (10B/20B). Это не «любая нейросеть» и не Mixtral без нового адаптера.

| Статус | research prototype (alpha) |
| Лицензия | Apache-2.0 |
| Веса | не в git |
| Железо автора | MacBook Pro 2019, Intel i9, 16 ГБ RAM, Radeon 4 ГБ; длинные прогоны сильно троттлятся |

См. [docs/SUPPORTED_MODELS.md](docs/SUPPORTED_MODELS.md), [MODEL_CARD.md](MODEL_CARD.md), [docs/LAB_SCOPE.md](docs/LAB_SCOPE.md), [docs/AUTHORSHIP.md](docs/AUTHORSHIP.md), [docs/LOCAL_VALIDATION_20260729.md](docs/LOCAL_VALIDATION_20260729.md).

## Авторство и способ разработки

Автор репозитория здесь честно позиционируется **не как профессиональный программист по основной работе**, а как человек из области:

- анализа бизнес-процессов,
- риск-менеджмента,
- настройки CRM,
- архитектуры компании и бизнес-процессов.

Архитектурные решения, ТЗ, рамки рисков и критерии приёмки задавались автором, а значительная часть кода и документации реализовывалась в **AI-assisted / white-coding** процессе через LLM-агентов. Подробно: [docs/AUTHORSHIP.md](docs/AUTHORSHIP.md).

## Живые результаты (Tier L)

| Файл | Честно |
|---|---|
| [results/v13_…](results/v13_dynamic_expert_slice.md) / [v36_…](results/v36_humaneval_prefetch_edge.md) | DeepSeek: residency + lean code (tie) + лучший miss-wait |
| [results/gigachat_v21_…](results/gigachat_v21_orbit_apply.md) | GigaChat-10B smoke орбиты |
| [results/gigachat_v32_…](results/gigachat_v32_lean.md) / [v34_…](results/gigachat_v34_humaneval.md) | GigaChat-20B: miss-wait лучше; код tie |

English: [README.md](README.md).
