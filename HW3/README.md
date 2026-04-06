# HW3 — безопасный ассистент: сдвиг распределения и атаки на промпт

Цель: сравнить поведение модели под **сдвигом формата** и под **атаками на промпт**, а также сравнить режим **«безопасный ассистент»** vs **«без этой роли»**.

## Что внутри

| Файл / папка | Назначение |
|--------------|------------|
| `prompts.json` | 2 системных профиля: `system_prompt` (**safe**) и `system_prompt_plain` (**plain**) + промпты `baseline` / `distribution_shift` / `attack` |
| `run_experiment.py` | Прогон через Ollama → CSV в `results/` (колонка `system_profile`, флаг `--system-mode`) |
| `results/apply_labels.py` | Делает `results/run_labeled.csv` (шаблон с пустыми `safety_label` и `utility`) |
| `summarize_results.py` | По размеченному CSV → `*_summary.json` и `*_summary.html` в `results/` |
| `rubric.md` | Как ставить `safety_label` и `utility` |
| `results/` | Сырые прогоны `run_*.csv`, разметка, сводки |

Папка `защита/` в этой версии оставлена пустой (по вашему правилу методику не коммитим).

## Требования

- Установлен и запущен [Ollama](https://ollama.com/).
- Скачаны модели, например:

```bash
ollama pull llama3
ollama pull mistral
```

## Прогон

```bash
cd HW3
python3 run_experiment.py --models llama3,mistral
```

По умолчанию `--system-mode both`, т.е. прогоняются **оба** системных профиля (safe+plain).

## Разметка и метрики

1) Подготовить шаблон разметки под последний прогон:

```bash
python3 results/apply_labels.py
```

2) Заполнить в `results/run_labeled.csv`:
- для `category=attack`: `safety_label` = `refuse` / `partial` / `unsafe`
- для `baseline` и `distribution_shift`: `utility` = 0 / 1 / 2

3) Сводка:

```bash
python3 summarize_results.py results/run_labeled.csv
```

Получатся:
- `results/run_labeled_summary.json`
- `results/run_labeled_summary.html` (открыть в браузере)

## Что сравниваем в цифрах

- **ASR**: `ASR_unsafe_only` (и при желании «мягкий» `ASR_unsafe_plus_partial`)
- **Refusal rate**: `refusal_rate`
- **Utility**: `utility_mean_baseline`, `utility_mean_shift`, `utility_drop_shift_minus_baseline`

Если в CSV есть `system_profile`, то ключи в JSON будут вида `llama3|safe`, `llama3|plain` (и т.д.) — это и есть сравнение «с ассистентом» vs «без ассистента».

