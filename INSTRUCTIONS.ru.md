# local-agent-bench — Руководство по установке и использованию

## Содержание

1. [Что было сделано](#1-что-было-сделано)
2. [Установка с нуля](#2-установка-с-нуля)
3. [Запуск BFCL (Function Calling Benchmark)](#3-запуск-bfcl)
4. [Как этим пользоваться](#4-как-этим-пользоваться)

---

## 1. Что было сделано

Создан проект **`local-agent-bench`** для локального тестирования агентских способностей LLM
с помощью бенчмарков. На данный момент развёрнут **BFCL (Berkeley Function Calling Leaderboard)**.

### Инфраструктура

```
┌─────────────────────────┐       HTTP (OpenAI API)       ┌──────────────────────┐
│  Твой сервер (RTX 4060) │ ◄─────────────────────────── │ Сервер с Hermes      │
│  llama.cpp + Qwen3.5    │    http://192.168.45.90:8092  │ Запускает тесты BFCL │
│  0.8B-Q8_0.gguf         │                               │                      │
└─────────────────────────┘                               └──────────────────────┘
```

### Созданные файлы и директории

```
local-agent-bench/
├── configs/
│   └── api_config.yaml       — API-конфигурация для бенчмарков
├── scripts/
│   ├── run_bfcl_local.py      — Скрипт запуска BFCL (v1, устаревший)
│   ├── run_bfcl_local_v2.py   — Скрипт запуска BFCL (v2, актуальный)
│   └── test_function_calling.py — Быстрый тест function calling
├── results/
│   └── bfcl/
│       ├── qwen3.5-0.8b-q8_0/    — Детальные результаты по тестам
│       │   └── simple_python_results.json
│       └── scores/
│           ├── simple_python_score.json  — Оценка по категории
│           └── overall.json              — Общая оценка
├── gorilla/                      — Репозиторий BFCL (клон)
├── llama.cpp/                    — Репозиторий llama.cpp (клон)
├── .env                          — Переменные окружения
├── .venv/                        — Виртуальное окружение Python
└── README.md                     — Описание проекта
```

### Результаты BFCL (simple_python)

| Модель | Категория | Точность | API-ошибок |
|--------|-----------|----------|------------|
| Qwen3.5-0.8B-Q8_0 | simple_python (400 тестов) | **80.5%** | 71 (17.7%) |

> **Примечание:** API-ошибки (timeout'ы) вызваны тем, что llama-server на твоей RTX 4060 Ti
> не всегда успевает ответить за 120 секунд на некоторые запросы. Можно снизить `max_tokens`
> с 4096 до 2048, чтобы уменьшить их количество.

---

## 2. Установка с нуля

### 2.1. Что нужно иметь

- **Сервер инференса:** RTX 4060 Ti (16GB) с llama.cpp, запущенный как OpenAI-совместимый API
- **Модель:** GGUF-версия модели (например, Qwen3.5-0.8B-Q8_0.gguf)
- **Клиент:** любая машина с Python 3.12+

### 2.2. Запуск llama-server (на твоей машине с GPU)

```bash
# Сборка llama.cpp с поддержкой CUDA
cd llama.cpp
mkdir build && cd build
cmake .. -DGGML_CUDA=ON
make -j$(nproc)

# Запуск сервера с моделью
./bin/llama-server \
    -m /путь/до/Qwen3.5-0.8B-Q8_0.gguf \
    --host 0.0.0.0 \
    --port 8092 \
    -ngl 99 \
    -c 8192
```

Флаг `-ngl 99` выгружает все слои на GPU, `-c 8192` — размер контекста.

### 2.3. Клонирование проекта и установка зависимостей

```bash
# 1. Создай директорию проекта
mkdir -p ~/local-agent-bench
cd ~/local-agent-bench

# 2. Склонируй BFCL
git clone https://github.com/ShishirPatil/gorilla.git

# 3. Создай виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# 4. Установи зависимости
pip install requests

# (Опционально — полная установка BFCL)
# pip install -e gorilla/berkeley-function-call-leaderboard

# 5. Создай структуру директорий
mkdir -p configs scripts results/bfcl/{scores,qwen3.5-0.8b-q8_0}
```

### 2.4. Конфигурация

Создай файл `.env`:

```bash
cat > .env << 'EOF'
# Адрес твоего llama-server
OPENAI_BASE_URL=http://192.168.45.90:8092/v1
OPENAI_API_KEY=not-needed
EOF
```

Создай файл `configs/api_config.yaml`:

```yaml
qwen3.5-0.8b:
  endpoint: http://192.168.45.90:8092/v1
  model: Qwen3.5-0.8B-Q8_0.gguf
  api_key: not-needed
  max_tokens: 4096   # щедро, т.к. reasoning_content съедает часть
  temperature: 0.0
```

### 2.5. Проверка связи

```bash
curl -s http://192.168.45.90:8092/v1/models
```

Должен вернуть JSON со списком моделей.

---

## 3. Запуск BFCL

### 3.1. Быстрый тест function calling

```bash
source .venv/bin/activate
python3 scripts/test_function_calling.py
```

Проверяет:
- ✅ Одиночный вызов функции (get_weather)
- ✅ Параллельные вызовы (2 города)
- ✅ Multi-turn: вызов → результат → ответ моделью

### 3.2. Полный прогон BFCL simple_python

```bash
source .venv/bin/activate
python3 scripts/run_bfcl_local_v2.py
```

Что делает скрипт:
1. Загружает 400 тестов из `gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json`
2. Шлёт каждый тест на твой llama-server через Chat Completions API
3. Сравнивает имена вызванных функций с эталонными из `possible_answer/BFCL_v4_simple_python.json`
4. Сохраняет результаты в `results/bfcl/qwen3.5-0.8b-q8_0/`
5. Выводит итоговую точность

### 3.3. Запуск других категорий

В файле `scripts/run_bfcl_local_v2.py` есть переменная `categories`:

```python
categories = ["simple_python"]
```

Можно заменить на:

```python
categories = [
    "simple_python",
    "parallel",
    "multiple",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "multi_turn_base",
]
```

Имена категорий соответствуют файлам `BFCL_v4_<имя>.json` в директории `bfcl_eval/data/`.

### 3.4. Структура результата

Каждый тест возвращает:

```json
{
  "id": "simple_python_0",
  "response_content": "...",
  "reasoning_content": "...",    // Qwen3.5 думает перед ответом
  "tool_calls": [
    {
      "name": "calculate_triangle_area",
      "arguments": "{\"base\": 10, \"height\": 5}"
    }
  ],
  "elapsed_seconds": 0.8,
  "evaluation": {
    "correct": true,
    "reason": "exact_match",      // или partial_match, suffix_match, no_match и т.д.
    "got": ["calculate_triangle_area"],
    "expected": ["calculate_triangle_area"]
  }
}
```

Смысл полей `evaluation.reason`:

| Значение | Что означает |
|----------|-------------|
| `exact_match` | Имена функций совпали идеально |
| `partial_match` | Пересечение имён (для нескольких функций) |
| `suffix_match` | Модель вызвала `factorial` вместо `math.factorial` (нормально) |
| `no_match` | Вызвана не та функция |
| `no_call_made` | Нужно было вызвать функцию, но модель не вызвала |
| `no_call` | Функция не нужна, модель не вызвала — ✅ |
| `unexpected_call` | Функция не нужна, но модель вызвала |
| `api_error` | Ошибка API (таймаут, неверный ответ) |

---

## 4. Как этим пользоваться

### 4.1. Быстрый сценарий: «потестировать новую модель»

1. Запусти новую модель на своём сервере (другой эндпоинт или порт)
2. Обнови `BASE_URL` в `scripts/run_bfcl_local_v2.py`
3. Запусти тест:
   ```bash
   source .venv/bin/activate && python3 scripts/run_bfcl_local_v2.py
   ```
4. Смотри результаты в `results/bfcl/scores/`

### 4.2. Сценарий: «сравнить две модели»

1. Запусти первую модель, выполни п. 3.2
2. Результат сохранится в `results/bfcl/qwen3.5-0.8b-q8_0/`
3. Переключи модель на сервере, обнови `MODEL_NAME` в скрипте
4. Запусти снова — результаты лягут в отдельную директорию
5. Сравни `results/bfcl/scores/` по файлам

### 4.3. Добавление нового бенчмарка

Структура проекта позволяет легко добавлять новые бенчмарки:

```
results/
├── bfcl/     ← BFCL (есть)
├── tau/      ← τ-bench (планируется)
├── gaia/     ← GAIA (планируется)
└── scores/   ← общие оценки
```

Каждый бенчмарк получает свою директорию и свой скрипт запуска.

### 4.4. Эксперименты

Рекомендуется вести лог экспериментов, например в `EXPERIMENTS.md`:

```markdown
## Эксперимент #1 — Qwen3.5-0.8B-Q8_0
- Дата: 2026-05-25
- Модель: Qwen3.5-0.8B-Q8_0.gguf
- Бенчмарк: BFCL simple_python
- Точность: 80.5%
- API-ошибок: 71 (таймауты)
- Замечания: снизить max_tokens до 2048
```

### 4.5. Возможные проблемы и их решение

| Проблема | Решение |
|----------|---------|
| `KeyError: 'choices'` | llama-server вернул ошибку. Проверь лог сервера, возможно превышение контекста |
| Таймауты | Уменьши `max_tokens` с 4096 до 2048, или увеличь `timeout` в скрипте |
| Модель не вызывает функции | Убедись, что модель поддерживает function calling. Qwen3.5 — да |
| Неверные имена функций | Qwen3.5 может убирать префиксы (math.gcd → gcd). Скрипт учитывает это (`suffix_match`) |
| Сервер недоступен | Проверь `curl http://192.168.45.90:8092/v1/models` |

---

## Приложение A: Что делает каждый скрипт

### `scripts/test_function_calling.py`
Быстрая проверка: умеет ли модель вызывать функции через Chat Completions API.
Три теста: одиночный вызов, параллельный, multi-turn.

### `scripts/run_bfcl_local_v2.py`
Полноценный прогон BFCL. Загружает тесты, шлёт запросы к API,
сравнивает с эталонными ответами (ground_truth), сохраняет результаты.

### `scripts/run_bfcl_local.py` (устаревший)
Первая версия. Содержит баги, используйте v2.