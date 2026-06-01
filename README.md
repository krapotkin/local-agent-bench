# local-agent-bench

Локальное тестирование агентских способностей LLM.

**Стек:** llama.cpp → OpenAI API → бенчмарки

## Инфраструктура

- **Сервер инференса:** `http://192.168.45.90:8092` — Qwen3.5-0.8B-Q8_0.gguf (RTX 4060 Ti)
- **Клиент (запуск тестов):** этот сервер (Hermes)

## Бенчмарки

- [ ] τ-bench — агентские диалоги (поддержка, бронирование)
- [ ] GAIA (Meta) — многошаговые real-world задачи
- [ ] BFCL — function calling
- [ ] SWE-bench — код-агенты

## Субмодули

| Модуль | Описание | Remote |
|---|---|---|
| `llama.cpp` | Локальный inference-движок (GGUF) | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| `gorilla` | Benchmarks: BFCL, agent-arena, OpenFunctions | [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) |

При клонировании обязательно подтягивай субмодули:
```bash
git clone --recurse-submodules https://github.com/<user>/local-agent-bench.git
```

Если репозиторий уже склонирован без субмодулей:
```bash
git submodule update --init --recursive
```

## Конфигурация API

См. `configs/api_config.yaml`