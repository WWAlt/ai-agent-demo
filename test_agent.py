"""Tests run without an API key: the LLM is faked."""
import json
from types import SimpleNamespace as NS
import agent as agent_mod
from agent import Agent
from tools import run_tool


def call(name, args, id="t1"):
    return NS(id=id, function=NS(name=name, arguments=json.dumps(args)),
              model_dump=lambda: {"id": id, "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args)}})


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.chat = NS(completions=self)
    def create(self, **kw):
        return NS(choices=[NS(message=NS(content="", tool_calls=self.replies.pop(0)))])


def test_tool_then_respond(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "MEMORY_FILE", tmp_path / "m.json")
    monkeypatch.setattr(agent_mod, "run_tool", lambda n, a: {"result": 42})
    a = Agent(FakeClient([
        [call("convert_currency", {"amount": 10, "from_currency": "EUR", "to_currency": "USD"})],
        [call("respond", {"status": "success", "answer": "10 EUR is 42 USD"}, "t2")],
    ]))
    r = a.ask("convert 10 eur to usd")
    assert r.status == "success" and r.tools_used == ["convert_currency"]
    assert len(a.history) == 2


def test_unknown_tool_is_handled():
    assert "error" in run_tool("nope", {})


def test_bad_model_output(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "MEMORY_FILE", tmp_path / "m.json")
    a = Agent(FakeClient([[call("respond", {"status": "weird"})]]))
    assert a.ask("hi").status == "error"
