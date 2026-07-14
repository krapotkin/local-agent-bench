# AGENTS.md — Контекст для AI-агента

## Структура проекта

```
local-agent-bench/
├── configs/api_config.yaml    — Конфиги моделей (endpoint, model, api_key, max_tokens, temperature)
├── data/gaia_validation.json  — GAIA validation dataset
├── scripts/
│   ├── run_bfcl_local_v2.py   — BFCL benchmark (function calling)
│   ├── run_gaia.py            — GAIA benchmark (ReAct agent loop)
│   └── test_function_calling.py — Быстрый тест function calling
├── run_benchmarks.sh          — Универсальный скрипт запуска
├── gorilla/                    — git submodule: ShishirPatil/gorilla (BFCL data)
└── llama.cpp/                  — git submodule: ggml-org/llama.cpp
```

## Результаты

Результаты хранятся в `~/workspace/data/local-agent-bench/benchmarks/`:

```
benchmarks/
├── bfcl/
│   ├── <model_name>/          — Результаты по модели
│   │   └── <category>_results.json
│   └── scores/
│       ├── <category>_score_<model>.json
│       └── overall.json
└── gaia/
    └── <model_name>/
        └── gaia_results.json
```

## BFCL данные

- Тесты: `gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_<category>.json`
- Ground truth: `gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_<category>.json`
- Формат ground truth: `{"id": "...", "ground_truth": [{"func_name": {"param": [vals]}}]}`
- Скрипт нормализует `ground_truth` → `[func_name, ...]` для сравнения

## GAIA данные

- Тесты: `data/gaia_validation.json` (локальная копия)
- Формат: JSON lines с полями `task_id`, `Question`, `Answer`, `Level`
- ReAct loop с инструментами: `execute_python`, `web_search`

## Скрипт run_bfcl_local_v2.py

- Поддерживает несколько моделей через `--model <key>`
- Конфиги в `configs/api_config.yaml`
- Результаты сохраняются прогрессивно (каждые 50 тестов)
- Поддерживает оба формата ground truth: `possible_answer` (старый) и `ground_truth` (новый)

## Скрипт run_gaia.py

- ReAct agent loop с лимитом шагов (по умолч. 10)
- Инструменты: `execute_python` (локальное выполнение), `web_search` (DuckDuckGo)
- Результаты сохраняются в `~/workspace/data/local-agent-bench/benchmarks/gaia/`

## Известные ограничения

1. **Multi-turn** (multi_turn_*, memory) — пока не поддерживается. Требуется логика нескольких туров диалога.
2. **Irrelevance** — нет ground truth, поэтому точность низкая (модель вызывает функции, а ground truth пустой).
3. **vLLM** — требует `--enable-auto-tool-choice --tool-call-parser <parser>` для function calling.
4. **llama.cpp** — работает без доп. флагов, но медленнее (таймауты на слабых GPU).

## Как добавить новую модель

1. Добавить секцию в `configs/api_config.yaml`
2. Проверить `curl <endpoint>/v1/models`
3. Запустить `./run_benchmarks.sh bfcl --model <key> --categories simple_python`

## Венв

- Венв: `~/workspace/venvs/local-agent-bench/default`
- Активация: `source .venv`
- Зависимости: `requests`, `pyyaml`, `duckduckgo-search`, `pandas`, `modelscope`
