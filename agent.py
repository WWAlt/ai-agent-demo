"""A lightweight AI agent: LLM + tools + memory + structured JSON output."""
import json
import os
import sys
from pathlib import Path
from typing import Literal, Optional

from openai import BadRequestError, OpenAI, OpenAIError
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from tools import TOOL_SPECS, run_tool

load_dotenv()
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
BASE_URL = "https://api.groq.com/openai/v1"
MEMORY_FILE = Path("memory.json")
MAX_TURNS_KEPT = 20   # keep context small
MAX_STEPS = 6         # max tool calls per request


class AgentResponse(BaseModel):
    """The structured answer every request returns."""
    status: Literal["success", "error", "needs_clarification"]
    answer: str
    tools_used: list[str] = []
    data: Optional[dict] = None


# The model must finish by calling this "tool" -> guarantees structured output
RESPOND_TOOL = {
    "name": "respond",
    "description": "Give the final answer to the user. Always finish with this.",
    "input_schema": {"type": "object", "properties": {
        "status": {"type": "string", "enum": ["success", "error", "needs_clarification"]},
        "answer": {"type": "string"},
        "data": {"type": "object"}},
        "required": ["status", "answer"]},
}


def to_openai(spec):
    return {"type": "function", "function": {
        "name": spec["name"], "description": spec["description"],
        "parameters": spec["input_schema"]}}

FINAL_NAMES = {"respond", "response", "final_answer", "answer", "final"}

SYSTEM = ("You are a helpful assistant. Use tools when you need live data. "
          "Use earlier conversation for context. If a tool returns an error, "
          "explain it plainly. Always finish by calling the `respond` tool.")


class Agent:
    def __init__(self, client=None):
        self.client = client or OpenAI(base_url=BASE_URL, max_retries=3,
                                      api_key=os.getenv("GROQ_API_KEY"))
        self.history = self._load()

    # ---- memory ----
    def _load(self) -> list:
        if MEMORY_FILE.exists():
            try:
                return json.loads(MEMORY_FILE.read_text())
            except json.JSONDecodeError:
                return []
        return []

    def _save(self):
        self.history = self.history[-MAX_TURNS_KEPT:]
        MEMORY_FILE.write_text(json.dumps(self.history, indent=2))

    def reset(self):
        self.history = []
        MEMORY_FILE.unlink(missing_ok=True)

    @staticmethod
    def _rescue(err: BadRequestError):
        """If the rejected generation was really a final answer, recover it."""
        try:
            gen = err.body.get("failed_generation") if isinstance(err.body, dict) else None
            gen = gen or (err.body or {}).get("error", {}).get("failed_generation")
            data = json.loads(gen)
            if data.get("name") in FINAL_NAMES:
                args = data.get("arguments") or data.get("parameters") or {}
                return json.loads(args) if isinstance(args, str) else args
        except Exception:
            pass
        return None

    # ---- main loop ----
    def ask(self, user_input: str) -> AgentResponse:
        messages = [{"role": "system", "content": SYSTEM}, *self.history,
                    {"role": "user", "content": user_input}]
        tools = [to_openai(t) for t in [*TOOL_SPECS, RESPOND_TOOL]]
        tools_used: list[str] = []

        try:
            for _ in range(MAX_STEPS):
                try:
                    msg = self.client.chat.completions.create(
                        model=MODEL, max_tokens=1024, messages=messages,
                        tools=tools, tool_choice="required",
                    ).choices[0].message
                except BadRequestError as e:
                    # Model misnamed a tool (e.g. "response" instead of "respond").
                    rescued = self._rescue(e)
                    if rescued is None:
                        messages.append({"role": "user", "content":
                            "Your last tool call was invalid. Use only the tools provided "
                            "and finish by calling `respond`."})
                        continue
                    final_args = rescued
                else:
                    calls = msg.tool_calls or []
                    final = next((c for c in calls if c.function.name in FINAL_NAMES), None)
                    final_args = json.loads(final.function.arguments or "{}") if final else None
                if final_args is not None:
                    resp = AgentResponse(**final_args)
                    resp.tools_used = tools_used
                    self.history += [{"role": "user", "content": user_input},
                                     {"role": "assistant", "content": resp.answer}]
                    self._save()
                    return resp

                messages.append({"role": "assistant", "content": msg.content or "",
                                 "tool_calls": [c.model_dump() for c in calls]})
                for c in calls:
                    tools_used.append(c.function.name)
                    try:
                        args = json.loads(c.function.arguments or "{}")
                        out = run_tool(c.function.name, args)
                    except json.JSONDecodeError:
                        out = {"error": "Invalid tool arguments"}
                    messages.append({"role": "tool", "tool_call_id": c.id,
                                     "content": json.dumps(out)})

            return AgentResponse(status="error", tools_used=tools_used,
                                 answer="Too many steps, please simplify the request.")
        except (ValidationError, json.JSONDecodeError) as e:
            return AgentResponse(status="error", answer=f"Invalid model output: {e}")
        except OpenAIError as e:
            return AgentResponse(status="error", tools_used=tools_used,
                                 answer=f"LLM API error: {e}")


def main():
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Missing GROQ_API_KEY. Copy .env.example to .env and add your key.")
    agent = Agent()
    print("AI Agent ready. Type 'reset' to clear memory, 'quit' to exit.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in ("quit", "exit"):
            break
        if text == "reset":
            agent.reset(); print("Memory cleared.\n"); continue
        if text:
            print(agent.ask(text).model_dump_json(indent=2), "\n")


if __name__ == "__main__":
    main()
