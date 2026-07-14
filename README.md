# local-agent-bench

Локальное тестирование агентских способностей LLM.

**Стек:** Любой OpenAI-compatible сервер → бенчмарки

## Бенчмарки

- [x] BFCL — function calling (Berkeley Function Calling Leaderboard)
- [x] GAIA — многошаговые real-world задачи (ReAct agent loop)
- [ ] tau-bench — агентские диалоги
- [ ] SWE-bench — код-агенты

## Быстрый старт

```bash
# 1. Подключи модель в configs/api_config.yaml
# 2. Проверь function calling
./run_benchmarks.sh test

# 3. Запусти BFCL
./run_benchmarks.sh bfcl --model <model_key> --categories simple_python

# 4. Запусти GAIA
./run_benchmarks.sh gaia --model <model_key>
```

## Результаты

### Gemma-4-12B-it-qat-w4a16-ct (vLLM)

#### BFCL

| Категория | Тестов | Точность |
|-----------|--------|----------|
| simple_python | 400 | **100.0%** |
| parallel_multiple | 200 | **100.0%** |
| parallel | 200 | **99.5%** |
| multiple | 200 | **98.5%** |
| simple_java | 100 | **88.0%** |
| simple_javascript | 50 | **88.0%** |
| irrelevance | 240 | 19.6% |
| multi_turn_base | 200 | 0.0%* |
| multi_turn_miss_func | 200 | 0.0%* |
| multi_turn_miss_param | 200 | 0.0%* |
| memory | 155 | 0.0%* |
| **Без multi_turn/memory** | **1390** | **84.5%** |

* Multi-turn и memory требуют отдельной логики (несколько туров диалога) — пока не поддерживаются скриптом.

#### GAIA

| Тестов | Точность |
|--------|----------|
| 100 | **19.0%** |

### Qwen3.5-0.8B-Q8_0 (llama.cpp, RTX 4060 Ti)

| Бенчмарк | Категория | Тестов | Точность |
|----------|-----------|--------|----------|
| BFCL | simple_python | 400 | 80.5% |

## Субмодули

| Модуль | Описание | Remote |
|--------|----------|--------|
| `llama.cpp` | Локальный inference-движок (GGUF) | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| `gorilla` | Benchmarks: BFCL, agent-arena, OpenFunctions | [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) |

## Конфигурация API

См. `configs/api_config.yaml`

## Детали

- **INSTALL.md** — установка, подключение моделей, запуск
- **AGENTS.md** — контекст для AI-агента
- **DATA.md** — описание данных и результатов
