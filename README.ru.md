# moe-orbit-prefetch (RU)

**Объект A — полностью открытый исходник:** орбита экспертов + sparse runtime DeepSeek с prefetch.

Лицензия кода: **MIT**. Веса моделей **не** кладём в git (лицензия DeepSeek/HF).

В репозитории есть всё нужное для использования объекта A:
- `OrbitPredictor`, `DynamicExpertStore`
- полный `SparseDeepseekRuntime` (generate без `from_pretrained` всей MoE)
- `deepseek_chat_engine.ask`
- smokes v13 и lean HumanEval/QuixBugs бенч, который мы реально гоняли

```bash
pip install -e ".[runtime]"
python examples/01_toy_orbit_no_weights.py
python examples/04_chat_ask.py "Привет" --max-new 32
```

Объект B (топология чата / RLM) сюда не входит. Граница утверждений: [WHAT_WE_CLAIM.md](WHAT_WE_CLAIM.md).
