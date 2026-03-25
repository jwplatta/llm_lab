Python Workspace
================

Use `python/src/` for modules and `python/tests/` for the accompanying tests. Run `uv sync` to ensure the virtual environment matches `pyproject.toml` and `uv.lock`.

### First experiment

`python/src/llm_lab/claude_agent.py` contains the 200-line toy Claude-style coding agent (tools for reading, listing, and editing files plus the conversational loop); run it once you populate `ANTHROPIC_API_KEY` in your `.env`.
