#!/usr/bin/env python3
"""
Run GAIA evaluation against a local OpenAI-compatible server.
Uses a simple ReAct agent loop with Python execution tools.
"""
import json
from typing import Any
import os
import sys
import time
import subprocess
import yaml
import requests
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "api_config.yaml"
DATA_PATH = PROJECT_ROOT / "data" / "gaia_validation.json"
RESULTS_DIR = Path.home() / "workspace" / "data" / "local-agent-bench" / "benchmarks" / "gaia"

# System prompt for the agent
SYSTEM_PROMPT = """You are an AI assistant designed to solve complex tasks.
You have access to the following tools:

1. `execute_python(code)`: Executes Python code and returns the output. Use this for calculations, data processing, etc.
2. `web_search(query)`: Searches the web for information. Use this for factual questions.

When you receive a tool response, analyze it and decide if you need more tools or if you can provide the final answer.
If you have the answer, respond with just the answer, no extra text.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code and return the stdout output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute."
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)

def execute_python_tool(code):
    """Safely execute Python code in a subprocess."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        return result.stdout if result.stdout else "(No output)"
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out."
    except Exception as e:
        return f"Error: {str(e)}"

def web_search_tool(query):
    """Mock web search (since we might not have internet)."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "\n".join([r["body"] for r in results])
            return "No results found."
    except Exception:
        return "Error: Web search unavailable (no internet or library issue)."

def run_agent(model_config, question, max_turns=10):
    """Run the agent loop for a single question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    
    headers = {"Authorization": f"Bearer {model_config.get('api_key', 'not-needed')}"}
    endpoint = model_config["endpoint"]
    
    for turn in range(max_turns):
        try:
            payload = {
                "model": model_config["model"],
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "max_tokens": model_config.get("max_tokens", 4096),
                "temperature": 0.0,
            }
            
            resp = requests.post(f"{endpoint}/chat/completions", json=payload, headers=headers, timeout=300)
            data = resp.json()
            
            if "choices" not in data:
                return None, f"API Error: {data.get('error', 'Unknown')}"
            
            msg = data["choices"][0]["message"]
            
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tool_name = tc["function"]["name"]
                    tool_args = json.loads(tc["function"]["arguments"])
                    
                    if tool_name == "execute_python":
                        tool_result = execute_python_tool(tool_args["code"])
                    elif tool_name == "web_search":
                        tool_result = web_search_tool(tool_args["query"])
                    else:
                        tool_result = "Unknown tool."
                    
                    messages.append(msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tool_name,
                        "content": tool_result
                    })
            else:
                # Final answer
                content = msg.get("content") or ""
                return content.strip(), "Success"
                
        except Exception as e:
            import traceback
            return None, f"Exception: {str(e)}\n{traceback.format_exc()}"
            
    return None, "Max turns reached"

def evaluate_answer(model_answer, ground_truth):
    """Simple string matching with some normalization."""
    if not model_answer:
        return False
    
    # Normalize: remove commas, convert to float if possible
    def normalize(s):
        s = s.replace(",", "").strip()
        try:
            return str(float(s))
        except:
            return s.lower()
            
    # Check exact match or substring
    norm_ans = normalize(model_answer)
    norm_gt = normalize(ground_truth)
    
    if norm_ans == norm_gt:
        return True
    
    # Check if model answer contains the ground truth or vice versa
    if norm_gt in norm_ans or norm_ans in norm_gt:
        return True
        
    return False

def main():
    config = load_config()
    # Default to first model or allow override
    model_key = sys.argv[1] if len(sys.argv) > 1 else list(config.keys())[0]
    
    if model_key not in config:
        print(f"Model {model_key} not found in config.")
        return
        
    mc = config[model_key]
    data = load_data()
    
    safe_model = model_key.replace("/", "-").strip("-")
    output_file = RESULTS_DIR / f"{safe_model}_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    correct = 0
    
    print(f"Running GAIA benchmark with model: {model_key}")
    print(f"Questions: {len(data)}")
    print("-" * 50)
    
    for i, item in enumerate(data):
        q_id = item["id"]
        question = item["Question"]
        gt = item["Answer"]
        
        print(f"[{i+1}/{len(data)}] {q_id}...", end=" ", flush=True)
        
        start_time = time.time()
        answer, status = run_agent(mc, question)
        elapsed = time.time() - start_time
        
        is_correct = evaluate_answer(answer, gt)
        if is_correct:
            correct += 1
            print(f"OK ({elapsed:.1f}s)")
        else:
            ans_preview = (answer or "None")[:50]
            print(f"FAIL ({elapsed:.1f}s) - Got: {ans_preview}...")
            
        results.append({
            "id": q_id,
            "question": question,
            "ground_truth": gt,
            "model_answer": answer,
            "correct": is_correct,
            "status": status,
            "elapsed_seconds": round(elapsed, 2)
        })
        
    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("-" * 50)
    print(f"Results: {correct}/{len(data)} = {correct/len(data)*100:.1f}%")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
