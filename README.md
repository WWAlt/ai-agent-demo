# Lightweight Python AI Agent

A small, reliable AI agent that:
- understands natural-language requests (Groq LLM, OpenAI-compatible)
- calls external APIs as tools (live weather, currency conversion)
- remembers the conversation (saved to `memory.json`)
- always returns **structured JSON** validated with Pydantic
- handles errors: API retries, timeouts, bad tool input, invalid model output

## Setup (macOS)
```bash
cd ai-agent-demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then paste your Groq API key into .env
python agent.py
```

## Example
```
You: What's the weather in Berlin, and how much is 50 EUR in USD?
{
  "status": "success",
  "answer": "It's 12°C in Berlin with light wind. 50 EUR is about 58 USD.",
  "tools_used": ["get_weather", "convert_currency"],
  "data": {...}
}
You: And in Tokyo?          <- uses memory: knows you mean the weather
```
Commands: `reset` clears memory, `quit` exits.

## Tests (no API key needed)
```bash
pytest -q
```

## How it works
1. User input + recent history go to the LLM with the tool list.
2. The LLM calls tools; results (or errors) are fed back.
3. The LLM must finish with the `respond` tool, so the output always matches the `AgentResponse` schema.
4. The turn is saved to memory (last 20 messages kept).

Add a new tool: write a function in `tools.py`, add its schema to `TOOL_SPECS` and `TOOLS`.
