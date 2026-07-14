# DATA.md — Описание данных проекта local-agent-bench

## ~/workspace/data/local-agent-bench/benchmarks/

Результаты бенчмарков, сохранённые после запуска скриптов.

### Структура

```
benchmarks/
├── bfcl/
│   ├── <model_name>/
│   │   └── <category>_results.json   — Детальные результаты по тестам
│   └── scores/
│       ├── <category>_score_<model>.json — Оценка по категории
│       └── overall.json              — Общая оценка
└── gaia/
    └── <model_name>/
        └── gaia_results.json         — Результаты GAIA
```

### Источники данных

- **BFCL** — Berkeley Function Calling Leaderboard, данные из субмодуля `gorilla/`.
- **GAIA** — GAIA benchmark от Meta, данные из `data/gaia_validation.json` (локальная копия).

### Форматы

- Результаты — JSON, один объект на тест с полями: `id`, `tool_calls`, `evaluation`, `elapsed_seconds`.
- Оценки — JSON с полями: `category`, `correct`, `total`, `accuracy`, `model`.
