# moe-orbit-prefetch (RU)

**Объект A — полный открытый исходник** (не урезанный дамп): орбита экспертов MoE + sparse runtime + **вся математика** в [docs/MATH.md](docs/MATH.md).

## Лицензия

- **Apache-2.0** — можно брать, править, встраивать. Денег автор не берёт.
- Мелкие эксперименты: достаточно соблюдать Apache (оставить LICENSE/NOTICE).
- Если метод уходит в **крупную систему/продукт** — укажите происхождение (этот репозиторий / метод orbit-prefetch). Подробности: [ATTRIBUTION.md](ATTRIBUTION.md).

Веса моделей в git **не** кладём (лицензия DeepSeek/HF). Код и формулы — да.

## Карта исходников

См. [docs/SOURCE_MANIFEST.md](docs/SOURCE_MANIFEST.md).

```bash
pip install -e ".[runtime]"
python examples/01_toy_orbit_no_weights.py
python examples/04_chat_ask.py "Привет" --max-new 32
```
