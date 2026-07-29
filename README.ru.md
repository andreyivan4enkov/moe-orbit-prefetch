# moe-orbit-prefetch

**Объект A — динамическая резидентность экспертов MoE по орбите из embedding / residual `h`.**

Простыми словами: не держим всю MoE в RAM. По `h` угадываем, какие эксперты понадобятся → подгружаем их куски из safetensors → после живого gate делаем deposit → холодных усыпляем (sleep/evict).

Репозиторий публикует **только объект A**. Память чата / RLM (объект B) сюда **не входит**.

| Документ | Зачем |
|---|---|
| [WHAT_WE_CLAIM.md](WHAT_WE_CLAIM.md) | Граница утверждений |
| [RELATED_WORK.md](RELATED_WORK.md) | Чужое / prior art |
| [results/](results/) | Санитизированные отчёты лаборатории |

## Что проверено

- v13: эксперт грузится куском; hit; sleep режет resident.
- v36: качество кода = classic; ожидание экспертов меньше; «обучение к концу растёт» — нет.

Не заявляем SOTA и не говорим, что предиктор = gate DeepSeek.

## Примеры

```bash
pip install -e ".[examples]"
python examples/01_toy_orbit_no_weights.py          # без весов
python examples/02_smoke_expert_slice.py            # нужен HF-кэш DeepSeek-V2-Lite-Chat
```

Веса моделей в репозиторий **не кладём** — только код (MIT).
