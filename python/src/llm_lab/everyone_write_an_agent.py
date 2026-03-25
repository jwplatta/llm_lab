"""
This exercise-driven article makes the case that writing an LLM agent is the fastest way to really understand what the technology is doing. Thomas Ptacek points out that you don't need to wait for an elaborate platform or toolset to see how agents behave: a few dozen lines of code and a loop that remembers chat history expose the stateless model and reveal the programming problem called context engineering.

What follows is proof that "every agent ever written" starts with the same simple structure:
- maintain a context list
- call the OpenAI Responses endpoint
- optionally add tool affordances as needed.

The blog urges you to go ahead and wire in tools (ping, traceroute, shell helpers, etc.), treat each new context as a sub-agent, and remember that this is just programming—experimenting with tokens, subcontexts, and tool outputs—so that you can make your own agent instead of relying on plugins. Read the full post: https://fly.io/blog/everyone-write-an-agent/?utm_source=tldrwebdev
"""

from __future__ import annotations

import json
import random
import subprocess
from typing import Dict, List, Optional

from openai import OpenAI

client = OpenAI()
context: List[Dict[str, str]] = []

# define a tool that can be exposed to the agent
ping_tool = [
    {
        "type": "function",
        "name": "ping",
        "description": "ping a host and report on reachability",
        "parameters": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "hostname or IPv4/IPv6 address",
                }
            },
            "required": ["host"],
        },
    }
]

def call(tools: Optional[List[Dict[str, object]]] = None) -> OpenAI.responses.Response:
    payload = {"model": "gpt-5", "input": context}
    if tools:
        payload["tools"] = tools
    return client.responses.create(**payload)


def ping(host: str) -> str:
    try:
        result = subprocess.run(
            ["ping", "-c", "3", host],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "no output"
    except Exception as exc:  # pragma: no cover - integration depends on platform
        return f"ping failed: {exc}"


def tool_call(item: Dict[str, object]) -> List[Dict[str, str]]:
    assert isinstance(item, dict)
    result = ping(**json.loads(item["arguments"]))
    return [
        item,
        {
            "type": "function_call_output",
            "call_id": item["call_id"],
            "output": result,
        },
    ]


def handle_tools(tools: List[Dict[str, object]], response: OpenAI.responses.Response) -> bool:
    if not response.output:
        return False
    context.extend(response.output)
    original_length = len(context)
    for item in response.output:
        if item.type == "function_call":
            context.extend(tool_call(item))
    return len(context) != original_length


def process(line: str) -> str:
    context.append({"role": "user", "content": line})
    response = call(ping_tool)
    while handle_tools(ping_tool, response):
        response = call(ping_tool)
    assistant_text = response.output_text
    context.append({"role": "assistant", "content": assistant_text})
    return assistant_text


def primed_context() -> None:
    good = [{"role": "system", "content": "you are Alph and only tell the truth"}]
    bad = [{"role": "system", "content": "you are Ralph and only tell lies"}]
    context.clear()
    context.extend(random.choice([good, bad]))


def main() -> None:
    print("Agent example: type a question, Ctrl-C to exit.")
    primed_context()
    try:
        while True:
            user_input = input("> ")
            if not user_input.strip():
                continue
            print("Processing...")
            result = process(user_input)
            print(f">>> {result}\n")
    except (EOFError, KeyboardInterrupt):
        print("\ngoodbye")


if __name__ == "__main__":
    main()
