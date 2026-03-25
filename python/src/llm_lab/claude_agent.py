"""
Toy Claude-style coding agent inspired by
https://www.mihaileric.com/The-Emperor-Has-No-Clothes/?utm_source=tldrdev.
The implementation mirrors the “read → list → edit” tool loop described
in the post: each step is documented in the dunder docstring below and the loop
simply proxies the LLM's tool calls back to the filesystem with light tracing.
"""

import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import anthropic
from dotenv import load_dotenv

load_dotenv()


def _get_claude_client() -> anthropic.Client:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY before running the toy Claude agent.")
    return anthropic.Client(api_key=api_key)

YOU_COLOR = "\u001b[94m"
ASSISTANT_COLOR = "\u001b[93m"
RESET_COLOR = "\u001b[0m"


def resolve_abs_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def read_file_tool(filename: str) -> Dict[str, Any]:
    """
    Read the contents of a file.
    :param filename: Path to the file that the LLM wants to inspect.
    :return: file_path and contents that will be sent back to Claude.
    """
    full_path = resolve_abs_path(filename)
    content = full_path.read_text(encoding="utf-8")
    return {"file_path": str(full_path), "content": content}


def list_files_tool(path: str) -> Dict[str, Any]:
    """
    List the entries in a directory so Claude knows what files exist.
    :param path: Directory whose listing Claude requested.
    :return: The canonical path plus simple metadata for each entry.
    """
    full_path = resolve_abs_path(path)
    all_files = []
    for item in full_path.iterdir():
        all_files.append({"filename": item.name, "type": "file" if item.is_file() else "dir"})
    return {"path": str(full_path), "files": all_files}


def edit_file_tool(path: str, old_str: str, new_str: str) -> Dict[str, Any]:
    """
    Replace text or create a new file when the assistant requests it.
    :param path: Path to the file to edit or create.
    :param old_str: String to replace (empty => create file).
    :param new_str: Replacement string that will be written.
    :return: Metadata describing what happened.
    """
    full_path = resolve_abs_path(path)
    if old_str == "":
        full_path.write_text(new_str, encoding="utf-8")
        return {"path": str(full_path), "action": "created_file"}
    original = full_path.read_text(encoding="utf-8")
    if old_str not in original:
        return {"path": str(full_path), "action": "old_str not found"}
    edited = original.replace(old_str, new_str, 1)
    full_path.write_text(edited, encoding="utf-8")
    return {"path": str(full_path), "action": "edited"}


TOOL_REGISTRY = {
    "read_file": read_file_tool,
    "list_files": list_files_tool,
    "edit_file": edit_file_tool,
}

# The blog emphasizes explaining tools clearly; each tool is surfaced through
# the system prompt so Claude can decide which one to call.


SYSTEM_PROMPT = """
You are an expert coding assistant that solves tasks by calling tools, one tool per line.
Use the tools described below exactly as documented. After each tool_result(...),
continue the conversation. When no tool is needed, reply normally.

{tool_list_repr}
"""


def get_tool_str_representation(tool_name: str) -> str:
    tool = TOOL_REGISTRY[tool_name]
    return (
        f"Name: {tool_name}\n"
        f"Description: {tool.__doc__.strip()}\n"
        f"Signature: {inspect.signature(tool)}"
    )


def get_full_system_prompt() -> str:
    tool_str_repr = ""
    for tool_name in TOOL_REGISTRY:
        tool_str_repr += "TOOL\n" + get_tool_str_representation(tool_name)
        tool_str_repr += "\n" + "=" * 30 + "\n"
    return SYSTEM_PROMPT.format(tool_list_repr=tool_str_repr)


def extract_tool_invocations(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    invocations: List[Tuple[str, Dict[str, Any]]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("tool:"):
            continue
        after = line[len("tool:"):].strip()
        if "(" not in after or not after.endswith(")"):
            continue
        name, rest = after.split("(", 1)
        name = name.strip()
        json_str = rest[:-1].strip()
        try:
            args = json.loads(json_str)
        except json.JSONDecodeError:
            continue
        invocations.append((name, args))
    return invocations


def execute_llm_call(conversation: List[Dict[str, str]]) -> str:
    response = _get_claude_client().chat.completions.create(
        model="claude-3.5-sonnet",
        messages=conversation,
        max_tokens_to_sample=2000,
    )
    return response.choices[0].message.content


def run_claude_agent_loop() -> None:
    print(get_full_system_prompt())
    conversation: List[Dict[str, str]] = [
        {"role": "system", "content": get_full_system_prompt()}
    ]

    while True:
        try:
            user_input = input(f"{YOU_COLOR}You:{RESET_COLOR} ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting toy Claude loop.")
            break
        conversation.append({"role": "user", "content": user_input.strip()})

        while True:
            # Each turn we let Claude respond; this mirrors the blog's inner loop
            assistant_response = execute_llm_call(conversation)
            tool_invocations = extract_tool_invocations(assistant_response)
            if not tool_invocations:
                print(f"{ASSISTANT_COLOR}Assistant:{RESET_COLOR} {assistant_response}")
                conversation.append({"role": "assistant", "content": assistant_response})
                break
            for name, args in tool_invocations:
                tool = TOOL_REGISTRY.get(name)
                if not tool:
                    print(f"Unknown tool: {name}")
                    continue
                print(f"{YOU_COLOR}Invoking:{RESET_COLOR} {name} {args}")
                if name == "read_file":
                    resp = tool(args.get("filename", "."))
                elif name == "list_files":
                    resp = tool(args.get("path", "."))
                else:
                    resp = tool(
                        args.get("path", "."),
                        args.get("old_str", ""),
                        args.get("new_str", ""),
                    )
                conversation.append(
                    {"role": "user", "content": f"tool_result({json.dumps(resp)})"}
                )


if __name__ == "__main__":
    run_claude_agent_loop()
