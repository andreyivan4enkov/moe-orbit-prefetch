# orbit_code_agent_v27

**verdict:** PASS_AGENT_ORBIT_EFFICIENCY

## Простыми словами

Код проходит цикл: контракт → генерация → реальный запуск проверок → один repair.
Добавлена защита памяти: при нехватке RAM холодные эксперты уходят в sleep,
резидентность сбрасывается перед каждой попыткой (в v26 процесс убила ОС).

stop=False
stop_reason=—

## Сводка

- ours: quality=1.000, full=1/1
- classic: quality=1.000, full=1/1
- comparisons={'quality_better': {'pass': False, 'gap': 0.0, 'mad': 2.220446049250313e-16, 'n': 1, 'wins': 0, 'losses': 0, 'win_frac': 0.0}, 'classic_quality_better': {'pass': False, 'gap': 0.0, 'mad': 2.220446049250313e-16, 'n': 1, 'wins': 0, 'losses': 0, 'win_frac': 0.0}, 'faster': {'pass': False, 'gap': -29.224160960002337, 'mad': 2.220446049250313e-16, 'n': 1, 'wins': 0, 'losses': 1, 'win_frac': 0.0}, 'less_miss_wait': {'pass': True, 'gap': 64.1472, 'mad': 2.220446049250313e-16, 'n': 1, 'wins': 1, 'losses': 0, 'win_frac': 1.0}}
