#!/usr/bin/env python3
"""Quick test: does Qwen3.5-0.8B understand function calling?"""

import json, requests

BASE = "http://192.168.45.10:8092/v1"

# Test: Simple function calling
# We ask the model to call get_weather for San Francisco
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. San Francisco"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

messages = [
    {"role": "user", "content": "What's the weather in San Francisco in fahrenheit?"}
]

print("=" * 60)
print("TEST: Function Calling via Chat Completions API")
print("=" * 60)

payload = {
    "model": "Qwen3.5-0.8B-Q8_0.gguf",
    "messages": messages,
    "tools": tools,
    "max_tokens": 2048,
    "temperature": 0.0,
}

resp = requests.post(f"{BASE}/chat/completions", json=payload, timeout=120)
data = resp.json()

msg = data["choices"][0]["message"]
print(f"Content: {msg.get('content', '')!r}")
print(f"Reasoning: {msg.get('reasoning_content', '')[:200]!r}")
if msg.get("tool_calls"):
    print(f"Tool calls: {json.dumps(msg['tool_calls'], indent=2)}")
else:
    print("❌ No tool calls returned")

print()
print("=" * 60)
print("TEST: Parallel function calls (get weather for 2 cities)")
print("=" * 60)

messages2 = [
    {"role": "user", "content": "What's the weather in San Francisco and Tokyo?"}
]

payload2 = {
    "model": "Qwen3.5-0.8B-Q8_0.gguf",
    "messages": messages2,
    "tools": tools,
    "max_tokens": 4096,
    "temperature": 0.0,
}

resp2 = requests.post(f"{BASE}/chat/completions", json=payload2, timeout=120)
data2 = resp2.json()

msg2 = data2["choices"][0]["message"]
print(f"Content: {msg2.get('content', '')!r}")
print(f"Reasoning preview: {msg2.get('reasoning_content', '')[:200]!r}")
if msg2.get("tool_calls"):
    for tc in msg2["tool_calls"]:
        print(f"  → Tool: {tc['function']['name']}, Args: {tc['function']['arguments']}")
else:
    print("❌ No tool calls returned")

print()
print("=" * 60)
print("TEST: Multi-turn function call (follow-up)")
print("=" * 60)

# Simulate: user asks, model calls tool, we return result, model responds
messages3 = [
    {"role": "user", "content": "What's the weather in London?"}
]

payload3 = {
    "model": "Qwen3.5-0.8B-Q8_0.gguf",
    "messages": messages3,
    "tools": tools,
    "max_tokens": 4096,
    "temperature": 0.0,
}

resp3 = requests.post(f"{BASE}/chat/completions", json=payload3, timeout=120)
data3 = resp3.json()
msg3 = data3["choices"][0]["message"]

if msg3.get("tool_calls"):
    # Simulate the tool result
    tool_call = msg3["tool_calls"][0]
    func_name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])
    
    messages3.append(msg3)  # assistant response with tool call
    messages3.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "name": func_name,
        "content": json.dumps({"temperature": 15, "condition": "Cloudy", "humidity": 72})
    })
    
    payload3b = {
        "model": "Qwen3.5-0.8B-Q8_0.gguf",
        "messages": messages3,
        "max_tokens": 4096,
        "temperature": 0.0,
    }
    
    resp3b = requests.post(f"{BASE}/chat/completions", json=payload3b, timeout=120)
    data3b = resp3b.json()
    msg3b = data3b["choices"][0]["message"]
    print(f"Final response: {msg3b.get('content', '')[:300]!r}")
else:
    print("❌ No tool calls in first turn")