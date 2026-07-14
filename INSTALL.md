# local-agent-bench — Руководство по установке и использованию

Локальное тестирование агентских способностей LLM с помощью бенчмарков.

## Содержание

1. [Что было сделано](#1-что-было-сделано)
2. [Установка с нуля](#2-установка-с-нуля)
3. [Подключение новой модели](#3-подключение-новой-модели)
4. [Запуск BFCL (Function Calling Benchmark)](#4-запуск-bfcl)
5. [Запуск GAIA (Agent Benchmark)](#5-запуск-gaia)
6. [Результаты](#6-результаты)
7. [Возможные проблемы](#7-возможные-проблемы)

---

## 1. Что было сделано

Проект **`local-agent-bench`** для локального тестирования агентских способностей LLM.
На данный момент развёрнуты **BFCL** (Berkeley Function Calling Leaderboard) и **GAIA** (Meta).

### Архитектура

```
┌──────────────────────────┐    HTTP (OpenAI API)     ┌─────────────────────┐
│  Сервер инференса        │ ◄───────────────────── │ Клиент (Hermes)     │
│  (vLLM / llama.cpp / …)  │    OpenAI-compatible    │ Запускает тесты BFCL │
│                          │                         │ Сохраняет результаты │
└──────────────────────────┘                         └─────────────────────┘
```

### Структура проекта

```
local-agent-bench/
├── configs/
│   └── api_config.yaml       — API-конфигурация для всех моделей
├── scripts/
│   ├── run_bfcl_local_v2.py  — BFCL benchmark
│   └── test_function_calling.py — Быстрый тест function calling
├── results/
│   └── bfcl/
│       ├── <model_name>/     — Детальные результаты по модели
│       │   └── <category>_results.json
│       └── scores/
│           ├── <category>_score_<model>.json
│           └── overall.json
├── gorilla/                   — Репозиторий BFCL (git submodule)
├── llama.cpp/                 — Репозиторий llama.cpp (git submodule)
├── .env                       — Переменные окружения
├── .venv                      — Bash-скрипт активации venv
├── README.md                  — Описание проекта
├── INSTALL.md                 — Эта инструкция
└── AGENTS.md                  — Контекст для AI-агента
```

### Поддерживаемые серверы инференса

| Сервер | Формат модели | Примечание |
|--------|--------------|------------|
| **vLLM** | HF-модель (автоматически) | `--enable-auto-tool-choice --tool-call-parser <parser>` обязателен |
| **llama.cpp** | GGUF | Работает из коробки, без доп. флагов |
| **ОГИ / любой OpenAI-compatible** | Зависит от сервера | Проверяй поддержку tool calling |

---

## 2. Установка с нуля

### 2.1. Требования

- **Сервер инференса** — любой OpenAI-compatible (vLLM, llama.cpp, OГИ и т.д.)
- **Клиент** — Python 3.12+ с `requests`, `pyyaml`

### 2.2. Клонирование

```bash
# Склонируй репозиторий с субмодулями
git clone --recurse-submodules <repo_url> local-agent-bench
cd local-agent-bench

# Если уже склонирован без субмодулей:
git submodule update --init --recursive
```

### 2.3. Виртуальное окружение

```bash
# Создай venv
python3 -m venv ~/workspace/venvs/local-agent-bench/default
source ~/workspace/venvs/local-agent-bench/default/bin/activate

# Установи зависимости
pip install requests pyyaml
```

### 2.4. Конфигурация модели

Добавь модель в `configs/api_config.yaml`:

```yaml
my-model:
  endpoint: http://<host>:<port>/v1
  model: <model_name_or_path>
  api_key: <key_or_not-needed>
  max_tokens: 4096
  temperature: 0.0
```

### 2.5. Проверка связи

```bash
curl -s http://<host>:<port>/v1/models
```

Должен вернуть JSON со списком моделей.

---

## 3. Подключение новой модели

### 3.1. Сервер на vLLM

**Важно:** для function calling обязательно запускай с флагами:

```bash
vllm serve /path/to/model \
    --host 0.0.0.0 \
    --port <port> \
    --enable-auto-tool-choice \
    --tool-call-parser <parser>
```

`--tool-call-parser` зависит от модели:
- `gemma` — для Gemma-серии
- `llama` — для Llama-серии
- `auto` — попытка автоопределения

Без этих флагов vLLM вернёт ошибку:
> `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`

### 3.2. Сервер на llama.cpp

```bash
./bin/llama-server \
    -m /path/to/model.gguf \
    --host 0.0.0.0 \
    --port <port> \
    -ngl 99 \
    -c 8192
```

Function calling работает из коробки — доп. флаги не нужны.

### 3.3. Сервер на OГИ (Open Gateway Interface)

Подключайся как к OpenAI-compatible API:

```yaml
my-model:
  endpoint: http://<host>:<port>/v1
  model: <model_name>
  api_key: <key>
  max_tokens: 4096
  temperature: 0.0
```

### 3.4. Быстрый тест function calling

```bash
source .venv
python3 scripts/test_function_calling.py
```

Проверяет: одиночный вызов, параллельные вызовы, multi-turn.

---

## 4. Запуск BFCL

### 4.1. Один прогон (simple_python)

```bash
source .venv
python3 scripts/run_bfcl_local_v2.py --model <model_key> --categories simple_python
```

### 4.2. Несколько категорий

```bash
python3 scripts/run_bfcl_local_v2.py \
    --model <model_key> \
    --categories simple_python,parallel,multiple,irrelevance,multi_turn_base
```

### 4.3. Все доступные категории

| Категория | Тестов | Ground truth | Описание |
|-----------|--------|-------------|----------|
| `simple_python` | 399 | [YES] | Базовые вызовы Python-функций |
| `simple_java` | 99 | [YES] | Java-функции |
| `simple_javascript` | 49 | [YES] | JavaScript-функции |
| `parallel` | 199 | [YES] | Параллельные вызовы |
| `multiple` | 199 | [YES] | Несколько функций в одном ответе |
| `parallel_multiple` | 199 | [YES] | Комбинация parallel + multiple |
| `irrelevance` | 239 | [NO] | Иррелевантные функции в списке |
| `multi_turn_base` | 200 | [YES] | Базовый multi-turn диалог |
| `multi_turn_long_context` | 200 | [NO] | Multi-turn с длинным контекстом |
| `multi_turn_miss_func` | 200 | [YES] | Пропуск функции в multi-turn |
| `multi_turn_miss_param` | 200 | [YES] | Пропуск параметра в multi-turn |
| `memory` | 155 | [YES] | Запоминание между турами |
| `format_sensitivity` | 9 | [NO] | Чувствительность к формату |
| `web_search` | 99 | [NO] | Web-поиск функции |
| `live_*` | ~1400 | [YES] (partial) | Live API-тесты (требуют интернет) |

### 4.4. Опции скрипта

```
--model <key>         Ключ модели из api_config.yaml (по умолч. первый)
--categories <list>   Запятые категории (по умолч. simple_python)
--max-tokens <N>      Переопределить max_tokens
--timeout <sec>       Таймаут запроса (по умолч. 180с)
```

### 4.5. Структура результата

```json
{
  "id": "simple_python_0",
  "response_content": "",
  "tool_calls": [
    {"name": "calculate_triangle_area", "arguments": "{\"base\":10,\"height\":5}"}
  ],
  "prompt_tokens": 357,
  "completion_tokens": 145,
  "elapsed_seconds": 0.85,
  "evaluation": {
    "correct": true,
    "reason": "exact_match",
    "got": ["calculate_triangle_area"],
    "expected": ["calculate_triangle_area"]
  }
}
```

### evaluation.reason

| Значение | Смысл |
|----------|-------|
| `exact_match` | Имена функций совпали идеально |
| `partial_match` | Пересечение имён (несколько функций) |
| `suffix_match` | `factorial` vs `math.factorial` (нормально) |
| `no_match` | Вызвана не та функция |
| `no_call_made` | Нужно было вызвать, но не вызвала |
| `no_call` | Функция не нужна, не вызвала — [YES] |
| `unexpected_call` | Функция не нужна, но вызвала |
| `api_error` / `timeout` | Ошибка сервера |
| `no_answer_data` | Нет ground truth для теста |

---

## 5. Запуск GAIA

GAIA — бенчмарк многошаговых real-world задач от Meta. Скрипт использует ReAct agent loop с инструментами Python-выполнения и web-поиска.

### 5.1. Быстрый запуск

```bash
source .venv
python3 scripts/run_gaia.py --model <model_key>
```

### 5.2. Что делает скрипт

1. Загружает тесты из `data/gaia_validation.json`
2. Для каждого теста запускает ReAct loop:
   - Модель получает задачу и список доступных инструментов
   - Модель решает: вызвать инструмент или дать финальный ответ
   - Если инструмент — результат возвращается модели для анализа
   - Цикл повторяется до финального ответа или лимита шагов
3. Сравнивает ответ модели с эталонным
4. Сохраняет результаты в `~/workspace/data/local-agent-bench/benchmarks/gaia/`

### 5.3. Доступные инструменты

| Инструмент | Описание |
|------------|----------|
| `execute_python(code)` | Выполняет Python-код, возвращает stdout |
| `web_search(query)` | Поиск в DuckDuckGo (требует интернет) |

### 5.4. Опции скрипта

```
--model <key>       Ключ модели из api_config.yaml (по умолч. первый)
--max-steps <N>     Максимум шагов ReAct loop (по умолч. 10)
--timeout <sec>     Таймаут запроса (по умолч. 300с)
```

---

## 6. Результаты

### Gemma-4-12B-it-qat-w4a16-ct (vLLM)

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

\* Multi-turn и memory требуют отдельной логики (несколько туров диалога) — пока не поддерживаются скриптом.

### Qwen3.5-0.8B-Q8_0 (llama.cpp, RTX 4060 Ti)

| Категория | Тестов | Точность | API-ошибок |
|-----------|--------|----------|------------|
| simple_python | 400 | 80.5% | 71 (17.7%) |

> API-ошибки вызваны таймаутами — llama.cpp на RTX 4060 Ti не всегда успевает за 120с.

---

## 7. Возможные проблемы

| Проблема | Решение |
|----------|---------|
| `vLLM: "auto" tool choice requires...` | Запусти сервер с `--enable-auto-tool-choice --tool-call-parser <parser>` |
| `KeyError: 'choices'` | Сервер вернул ошибку. Проверь лог, возможно превышение контекста |
| Таймауты | Уменьши `max_tokens` (4096 → 2048) или увеличь `--timeout` |
| Модель не вызывает функции | Убедись, что модель поддерживает function calling |
| Неверные имена функций | Скрипт учитывает суффиксы (`math.gcd` → `gcd`) |
| Сервер недоступен | Проверь `curl <endpoint>/v1/models` |
| `ModuleNotFoundError: yaml` | `pip install pyyaml` |
