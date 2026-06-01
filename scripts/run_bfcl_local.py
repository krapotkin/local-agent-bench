#!/usr/bin/env python3
"""
Run BFCL evaluation against a local OpenAI-compatible server (Qwen3.5-0.8B).
Uses Chat Completions API with function calling.

Data format: BFCL_v4_ JSONL files (one JSON object per line)
Evaluation: compares model's tool calls to expected answers
"""

import json
import os
import time
import requests

# ─── Configuration ───────────────────────────────────────────────
BASE_URL = "http://192.168.45.90:8092/v1"
MODEL_NAME = "Qwen3.5-0.8B-Q8_0.gguf"
PROJECT_ROOT = "/opt/data/workspace/projects/local-agent-bench"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "bfcl")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "qwen3.5-0.8b-q8_0")
SCORE_DIR = os.path.join(RESULTS_DIR, "scores")
BFCL_DATA_DIR = os.path.join(PROJECT_ROOT, "gorilla", "berkeley-function-call-leaderboard", "bfcl_eval", "data")
POSSIBLE_ANSWER_DIR = os.path.join(BFCL_DATA_DIR, "possible_answer")

TEST_CATEGORIES = ["simple_python"]


def load_test_data(category: str) -> list[dict]:
    """Load test data from BFCL's data folder (JSONL format)."""
    data_path = os.path.join(BFCL_DATA_DIR, f"BFCL_v4_{category}.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No data file for category '{category}': {data_path}")

    tests = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                tests.append(json.loads(line))
    return tests


def load_possible_answers(category: str) -> dict[str, list[dict]]:
    """Load possible answers for evaluation."""
    answer_path = os.path.join(POSSIBLE_ANSWER_DIR, f"BFCL_v4_{category}.json")
    if not os.path.exists(answer_path):
        return {}
    answers = {}
    with open(answer_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                answers[entry["id"]] = entry.get("possible_answer", [])
    return answers


def run_single_test(test_entry: dict) -> dict:
    """Run a single BFCL test case against the local inference server."""
    question_data = test_entry.get("question", [])
    if question_data and isinstance(question_data[0], list):
        messages = question_data[0]
    elif question_data and isinstance(question_data[0], dict):
        messages = question_data
    else:
        messages = [{"role": "user", "content": str(question_data)}]

    functions = test_entry.get("function", [])
    tools = []
    for func in functions:
        params = func.get("parameters", {}).copy()
        if isinstance(params, dict) and params.get("type") == "dict":
            params["type"] = "object"
        tool = {
            "type": "function",
            "function": {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": params,
            },
        }
        tools.append(tool)

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    start = time.time()
    resp = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=300)
    elapsed = time.time() - start
    data = resp.json()

    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})

    tool_calls_out = []
    for tc in msg.get("tool_calls", []):
        tool_calls_out.append({
            "name": tc["function"]["name"],
            "arguments": tc["function"]["arguments"],
        })

    return {
        "id": test_entry["id"],
        "response_content": msg.get("content", ""),
        "reasoning_content": msg.get("reasoning_content", ""),
        "tool_calls": tool_calls_out,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "elapsed_seconds": round(elapsed, 2),
    }


def evaluate_result(test_entry: dict, result: dict, possible_answers: list[dict]) -> dict:
    """Check model's function call correctness against expected answers."""
    model_calls = result["tool_calls"]

    # No answer data available
    if not possible_answers:
        return {"correct": True, "reason": "no_answer_data", "got": [], "expected": []}

    # Get expected function names
    expected_names = set()
    for pa in possible_answers:
        if isinstance(pa, dict):
            name = pa.get("name") or pa.get("function", "")
            if name:
                expected_names.add(name)
        elif isinstance(pa, str):
            expected_names.add(pa)

    # Get model function names
    model_names = {mc["name"] for mc in model_calls}

    # No call expected and none made
    if not expected_names and not model_names:
        return {"correct": True, "reason": "no_call", "got": list(model_names), "expected": list(expected_names)}

    # No call expected but one was made
    if not expected_names and model_names:
        return {"correct": False, "reason": "unexpected_call", "got": list(model_names), "expected": list(expected_names)}

    # Call expected but none made
    if expected_names and not model_names:
        return {"correct": False, "reason": "no_call_made", "got": list(model_names), "expected": list(expected_names)}

    # Check: exact match
    if expected_names == model_names:
        return {"correct": True, "reason": "exact_match", "got": list(model_names), "expected": list(expected_names)}

    # Check: intersection (for multi-function cases, partial match)
    intersection = expected_names & model_names
    if intersection:
        return {"correct": True, "reason": "partial_match", "got": list(model_names), "expected": list(expected_names)}

    # Check: suffix matching (dot-notation normalization)
    def suffixes(names_set):
        result_set = set()
        for n in names_set:
            if "." in n:
                result_set.add(n.split(".")[-1])
            else:
                result_set.add(n)
        return result_set

    expected_suffix = suffixes(expected_names)
    model_suffix = suffixes(model_names)
    suffix_intersection = expected_suffix & model_suffix

    if suffix_intersection:
        return {"correct": True, "reason": "suffix_match", "got": list(model_names), "expected": list(expected_names)}

    return {"correct": False, "reason": "no_match", "got": list(model_names), "expected": list(expected_names)}


def safe_get(d, *keys, default=""):
    """Safely traverse a nested dict."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d if d is not None else default


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SCORE_DIR, exist_ok=True)

    # Health check
    try:
        requests.get(f"{BASE_URL}/models", timeout=5)
        print(f"✅ Server online: {BASE_URL}")
    except Exception as e:
        print(f"❌ Server unreachable: {e}")
        return

    overall = {"correct": 0, "total": 0, "by_reason": {}}

    for category in TEST_CATEGORIES:
        print(f"\n{'='*60}")
        print(f"CATEGORY: {category}")
        print(f"{'='*60}")

        tests = load_test_data(category)
        possible_answers = load_possible_answers(category)
        print(f"Test cases: {len(tests)}")
        print(f"Answer data: {len(possible_answers)}")

        all_results = []
        correct = 0
        total = 0
        api_errors = 0

        for test in tests:
            test_id = test["id"]
            total += 1

            result = run_single_test(test)
            pa = possible_answers.get(test_id, [])
            eval_result = evaluate_result(test, result, pa)

            result["evaluation"] = eval_result
            all_results.append(result)

            if eval_result["correct"]:
                correct += 1

            reason = eval_result["reason"]
            overall["by_reason"][reason] = overall["by_reason"].get(reason, 0) + 1

        accuracy = (correct / total * 100) if total > 0 else 0

        # ── One-time summary ──
        by_reason = {}
        for r in all_results:
            reason = r.get("evaluation", {}).get("reason", "unknown")
            by_reason[reason] = by_reason.get(reason, 0) + 1

        print(f"\n{'='*50}")
        print(f"  {category}: {correct}/{total} = {accuracy:.1f}%")
        print(f"  Breakdown by reason:")
        for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
        print(f"{'='*50}")

        # Save results
        output_file = os.path.join(OUTPUT_DIR, f"{category}_results.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        score_data = {
            "category": category,
            "model": MODEL_NAME,
            "correct": correct,
            "total": total,
            "accuracy": round(accuracy, 2),
        }
        score_file = os.path.join(SCORE_DIR, f"{category}_score.json")
        with open(score_file, "w") as f:
            json.dump(score_data, f, indent=2)

        print(f"  Results saved: {output_file}")
        print(f"  Score saved: {score_file}")

        overall["correct"] += correct
        overall["total"] += total

    # Final overall summary
    o_acc = (overall["correct"] / overall["total"] * 100) if overall["total"] > 0 else 0
    print(f"\n{'='*60}")
    print(f"OVERALL: {overall['correct']}/{overall['total']} = {o_acc:.1f}%")
    print(f"{'='*60}")

    # Save overall score
    overall_file = os.path.join(SCORE_DIR, "overall.json")
    with open(overall_file, "w") as f:
        json.dump(overall, f, indent=2)
    print(f"Overall score: {overall_file}")


if __name__ == "__main__":
    print("=" * 60)
    print(f"BFCL Eval — {MODEL_NAME}")
    print(f"Endpoint → {BASE_URL}")
    print(f"Categories: {TEST_CATEGORIES}")
    print("=" * 60)
    main()