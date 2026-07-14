#!/usr/bin/env python3
"""
Run BFCL evaluation against a local OpenAI-compatible server.
Handles API errors gracefully. Saves results progressively.

Usage:
  python3 scripts/run_bfcl_local_v2.py [--model <model_key>] [--categories <cat1,cat2,...>]

Model keys are defined in configs/api_config.yaml.
Default: first model in config.
"""
import json, os, sys, time, argparse, yaml, requests
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "api_config.yaml")
DATA_DIR = os.path.join(PROJECT_ROOT, "gorilla", "berkeley-function-call-leaderboard", "bfcl_eval", "data")
ANSWER_DIR = os.path.join(DATA_DIR, "possible_answer")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Run BFCL benchmarks")
    parser.add_argument("--model", type=str, default=None, help="Model key from api_config.yaml")
    parser.add_argument("--categories", type=str, default="simple_python", help="Comma-separated category list")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max_tokens")
    parser.add_argument("--timeout", type=int, default=180, help="Request timeout in seconds")
    return parser.parse_args()


def get_model_config(config, model_key):
    if model_key and model_key in config:
        return model_key, config[model_key]
    # Default: first model
    key = list(config.keys())[0]
    return key, config[key]


def load_tests(category):
    path = os.path.join(DATA_DIR, f"BFCL_v4_{category}.json")
    tests = []
    with open(path) as f:
        for line in f:
            if line.strip():
                tests.append(json.loads(line))
    return tests


def load_answers(category):
    path = os.path.join(ANSWER_DIR, f"BFCL_v4_{category}.json")
    answers = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    # Support both old format (possible_answer) and new (ground_truth)
                    raw = entry.get("possible_answer") or entry.get("ground_truth", [])
                    # Normalize ground_truth format: [{"func_name": {"param": [vals]}}] -> [func_name, ...]
                    if raw and isinstance(raw, list) and raw and isinstance(raw[0], dict):
                        raw = [list(d.keys())[0] for d in raw if d]
                    answers[entry["id"]] = raw
    return answers


def run_test(test, mc, timeout):
    """Returns (result_dict, eval_dict)."""
    question_data = test.get("question", [])
    if question_data and isinstance(question_data[0], list):
        messages = question_data[0]
    elif question_data and isinstance(question_data[0], dict):
        messages = question_data
    else:
        messages = [{"role": "user", "content": str(question_data)}]

    functions = test.get("function", [])
    tools = []
    for func in functions:
        params = func.get("parameters", {}).copy()
        if isinstance(params, dict) and params.get("type") == "dict":
            params["type"] = "object"
        tools.append({
            "type": "function",
            "function": {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": params,
            },
        })

    payload = {
        "model": mc["model"],
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": mc.get("max_tokens", 4096),
        "temperature": mc.get("temperature", 0.0),
    }

    headers = {"Authorization": f"Bearer {mc.get('api_key', 'not-needed')}"}

    try:
        start = time.time()
        resp = requests.post(
            f"{mc['endpoint']}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        elapsed = time.time() - start
        data = resp.json()

        if "choices" not in data or not data["choices"]:
            err = data.get("error", {}).get("message", str(data))
            return {"error": f"API: {err[:200]}", "_eval": {"correct": False, "reason": "api_error", "got": [], "expected": []}}, None

        msg = data["choices"][0]["message"]
        usage = data.get("usage", {})

        tool_calls = []
        for tc in msg.get("tool_calls", []):
            tool_calls.append({
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
            })

        return {
            "response_content": msg.get("content", ""),
            "reasoning_content": msg.get("reasoning_content", ""),
            "tool_calls": tool_calls,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "elapsed_seconds": round(elapsed, 2),
        }, None

    except requests.Timeout:
        return {"error": "timeout", "_eval": {"correct": False, "reason": "timeout", "got": [], "expected": []}}, None
    except Exception as e:
        return {"error": str(e), "_eval": {"correct": False, "reason": "exception", "got": [], "expected": []}}, None


def evaluate(test_id, result, possible_answer):
    """Evaluate tool calls. Returns eval dict."""
    if "error" in result:
        return result.get("_eval", {"correct": False, "reason": "error", "got": [], "expected": []})

    pc = result["tool_calls"]
    if not possible_answer:
        return {"correct": bool(pc), "reason": "no_answer_data", "got": [c["name"] for c in pc], "expected": []}

    expected_names = set()
    for pa in possible_answer:
        if isinstance(pa, dict):
            name = pa.get("name") or pa.get("function", "")
            if name:
                expected_names.add(name)
        elif isinstance(pa, str):
            expected_names.add(pa)

    model_names = {c["name"] for c in pc}

    if not expected_names and not model_names:
        return {"correct": True, "reason": "no_call", "got": [], "expected": []}
    if not expected_names and model_names:
        return {"correct": False, "reason": "unexpected_call", "got": list(model_names), "expected": list(expected_names)}
    if expected_names and not model_names:
        return {"correct": False, "reason": "no_call_made", "got": [], "expected": list(expected_names)}
    if expected_names == model_names:
        return {"correct": True, "reason": "exact_match", "got": list(model_names), "expected": list(expected_names)}
    if expected_names & model_names:
        return {"correct": True, "reason": "partial_match", "got": list(model_names), "expected": list(expected_names)}

    def suffixes(ns):
        return {n.split(".")[-1] if "." in n else n for n in ns}

    if suffixes(expected_names) & suffixes(model_names):
        return {"correct": True, "reason": "suffix_match", "got": list(model_names), "expected": list(expected_names)}

    return {"correct": False, "reason": "no_match", "got": list(model_names), "expected": list(expected_names)}


def main():
    args = parse_args()
    config = load_config()
    model_key, mc = get_model_config(config, args.model)

    if args.max_tokens:
        mc["max_tokens"] = args.max_tokens

    categories = [c.strip() for c in args.categories.split(",")]

    # Sanitize model key for directory names
    safe_model = model_key.replace("/", "-").strip("-")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "bfcl", safe_model)
    SCORE_DIR = os.path.join(PROJECT_ROOT, "results", "bfcl", "scores")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SCORE_DIR, exist_ok=True)

    # Health check
    try:
        headers = {"Authorization": f"Bearer {mc.get('api_key', 'not-needed')}"}
        r = requests.get(f"{mc['endpoint']}/models", headers=headers, timeout=5)
        print(f"  Server: {mc['endpoint']}")
        print(f"  Model:  {mc['model']}")
        print(f"  OK")
    except Exception as e:
        print(f"  Server unreachable: {e}")
        return

    overall = {"correct": 0, "total": 0, "api_errors": 0}

    for category in categories:
        print(f"\n=== CATEGORY: {category} ===")
        tests = load_tests(category)
        answers = load_answers(category)
        print(f"  Tests: {len(tests)}, Answers: {len(answers)}")

        results = []
        correct = 0
        api_errors = 0
        total_time = 0.0

        for idx, test in enumerate(tests):
            test_id = test["id"]

            if (idx + 1) % 25 == 0 or idx == 0:
                avg_time = total_time / (idx + 1) if idx > 0 else 0
                eta = avg_time * (len(tests) - idx - 1)
                print(f"  [{idx+1}/{len(tests)}] {test_id} ... (ETA: {eta:.0f}s)", flush=True)

            result, _ = run_test(test, mc, args.timeout)

            ev = evaluate(test_id, result, answers.get(test_id, []))
            result["evaluation"] = ev
            result["id"] = test_id
            results.append(result)

            if "error" in result:
                api_errors += 1
            if ev["correct"]:
                correct += 1
            total_time += result.get("elapsed_seconds", 0)

            # Progressive save every 50 tests
            if (idx + 1) % 50 == 0:
                output_file = os.path.join(OUTPUT_DIR, f"{category}_results.json")
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

        # Summary
        by_reason = {}
        for r in results:
            reason = r.get("evaluation", {}).get("reason", "unknown")
            by_reason[reason] = by_reason.get(reason, 0) + 1

        avg_time = total_time / len(tests) if tests else 0
        print(f"\n--- {category} ---")
        print(f"  {correct}/{len(tests)} = {correct/len(tests)*100:.1f}%")
        print(f"  API errors: {api_errors}")
        print(f"  Avg time/test: {avg_time:.1f}s, Total: {total_time:.0f}s")
        for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

        # Save
        output_file = os.path.join(OUTPUT_DIR, f"{category}_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {output_file}")

        score_file = os.path.join(SCORE_DIR, f"{category}_score_{safe_model}.json")
        score = {
            "category": category,
            "model": model_key,
            "model_config": mc["model"],
            "correct": correct,
            "total": len(tests),
            "accuracy": round(correct / len(tests) * 100, 2) if tests else 0,
            "api_errors": api_errors,
            "avg_time_per_test": round(avg_time, 2),
            "total_time": round(total_time, 1),
        }
        with open(score_file, "w") as f:
            json.dump(score, f, indent=2)
        print(f"  Saved: {score_file}")

        overall["correct"] += correct
        overall["total"] += len(tests)
        overall["api_errors"] += api_errors

    print(f"\n=== OVERALL ===")
    o_acc = overall["correct"] / overall["total"] * 100 if overall["total"] else 0
    print(f"  {overall['correct']}/{overall['total']} = {o_acc:.1f}%")
    print(f"  API errors: {overall['api_errors']}")

    with open(os.path.join(SCORE_DIR, "overall.json"), "w") as f:
        json.dump(overall, f, indent=2)


if __name__ == "__main__":
    main()
