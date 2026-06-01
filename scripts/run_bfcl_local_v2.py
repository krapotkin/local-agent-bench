#!/usr/bin/env python3
"""
Run BFCL evaluation against a local OpenAI-compatible server.
Handles API errors gracefully. Saves results progressively.
"""
import json, os, time, requests

BASE_URL = "http://192.168.45.90:8092/v1"
MODEL_NAME = "Qwen3.5-0.8B-Q8_0.gguf"
PROJECT_ROOT = "/opt/data/workspace/projects/local-agent-bench"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "bfcl", "qwen3.5-0.8b-q8_0")
SCORE_DIR = os.path.join(PROJECT_ROOT, "results", "bfcl", "scores")
DATA_DIR = os.path.join(PROJECT_ROOT, "gorilla", "berkeley-function-call-leaderboard", "bfcl_eval", "data")
ANSWER_DIR = os.path.join(DATA_DIR, "possible_answer")

categories = ["simple_python"]


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
                    answers[entry["id"]] = entry.get("possible_answer", [])
    return answers


def run_test(test):
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
        "model": MODEL_NAME,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    try:
        start = time.time()
        resp = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=120)
        elapsed = time.time() - start
        data = resp.json()

        if "choices" not in data or not data["choices"]:
            err = data.get("error", {}).get("message", str(data))
            return {"error": f"API: {err[:200]}"}, {"correct": False, "reason": "api_error", "got": [], "expected": []}

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
        }, None  # None means eval not done yet

    except requests.Timeout:
        return {"error": "timeout"}, {"correct": False, "reason": "timeout", "got": [], "expected": []}
    except Exception as e:
        return {"error": str(e)}, {"correct": False, "reason": f"exception", "got": [], "expected": []}


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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SCORE_DIR, exist_ok=True)

    # Health check
    try:
        r = requests.get(f"{BASE_URL}/models", timeout=5)
        print(f"  ✅ Server: {BASE_URL}")
    except Exception as e:
        print(f"  ❌ Server unreachable: {e}")
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

        for idx, test in enumerate(tests):
            test_id = test["id"]

            if (idx + 1) % 25 == 0 or idx == 0:
                print(f"  [{idx+1}/{len(tests)}] {test_id} ...", flush=True)

            result, eval_result = run_test(test)
            if eval_result is not None:
                # Error case
                result["_eval"] = eval_result
                api_errors += 1

            ev = evaluate(test_id, result, answers.get(test_id, []))
            result["evaluation"] = ev
            result["id"] = test_id
            results.append(result)

            if ev["correct"]:
                correct += 1

        # Summary
        by_reason = {}
        for r in results:
            reason = r.get("evaluation", {}).get("reason", "unknown")
            by_reason[reason] = by_reason.get(reason, 0) + 1

        print(f"\n--- {category} ---")
        print(f"  {correct}/{len(tests)} = {correct/len(tests)*100:.1f}%")
        print(f"  API errors: {api_errors}")
        for reason, count in sorted(by_reason.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

        # Save
        output_file = os.path.join(OUTPUT_DIR, f"{category}_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  → {output_file}")

        score_file = os.path.join(SCORE_DIR, f"{category}_score.json")
        score = {"category": category, "model": MODEL_NAME,
                 "correct": correct, "total": len(tests),
                 "accuracy": round(correct/len(tests)*100, 2) if tests else 0,
                 "api_errors": api_errors}
        with open(score_file, "w") as f:
            json.dump(score, f, indent=2)
        print(f"  → {score_file}")

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