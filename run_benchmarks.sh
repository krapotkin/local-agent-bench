#!/usr/bin/env bash
# Запуск бенчмарков local-agent-bench
#
# Использование:
#   ./run_benchmarks.sh bfcl [--model <key>] [--categories <cat1,cat2>]
#   ./run_benchmarks.sh gaia [--model <key>]
#   ./run_benchmarks.sh test    — Быстрый тест function calling

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Активируем venv
source .venv

case "${1:-}" in
    bfcl)
        shift
        echo "=== BFCL Benchmark ==="
        python3 scripts/run_bfcl_local_v2.py "$@"
        ;;
    gaia)
        shift
        echo "=== GAIA Benchmark ==="
        python3 scripts/run_gaia.py "$@"
        ;;
    test)
        echo "=== Function Calling Test ==="
        python3 scripts/test_function_calling.py
        ;;
    *)
        echo "Использование: $0 {bfcl|gaia|test} [args...]"
        echo ""
        echo "Команды:"
        echo "  bfcl   — Запуск BFCL benchmark"
        echo "  gaia   — Запуск GAIA benchmark"
        echo "  test   — Быстрый тест function calling"
        exit 1
        ;;
esac
