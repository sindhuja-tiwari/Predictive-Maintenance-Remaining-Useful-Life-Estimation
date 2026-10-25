"""Copilot tools that work without an LLM key, Neo4j, or the RAG stack."""
import pytest

import data_prep as dp
from copilot import graph_kg, rul_tool


def test_predict_rul_tool_matches_service_contract():
    df = dp.load_raw("test")
    cycles = df[df.unit == 20].tail(20)[dp.SENSOR_COLS].to_dict("records")
    out = rul_tool.predict_rul(cycles, unit_id=20)
    assert set(out) >= {"predicted_rul_cycles", "health_index", "alert", "recommendation"}


def test_predict_rul_tool_reports_missing_sensors_instead_of_guessing():
    out = rul_tool.predict_rul([{"sensor_1": 520, "sensor_2": 640}] * 20)
    assert "error" in out and "sensor_3" in out["error"]


@pytest.mark.parametrize("name,mode", [
    ("vibration", "Bearing Wear"),
    ("Fan Speed", "Fan Module Degradation"),
    ("sensor_3", "HPC Degradation"),
    ("SENSOR_11", "Seal Leakage"),
])
def test_failure_modes_accepts_aliases(name, mode, monkeypatch):
    monkeypatch.setattr(graph_kg, "NEO4J_URI", None)
    assert graph_kg.failure_modes_for_sensor(name)[0]["mode"] == mode


def test_failure_modes_unknown_sensor(monkeypatch):
    monkeypatch.setattr(graph_kg, "NEO4J_URI", None)
    assert graph_kg.failure_modes_for_sensor("sensor_1") == []


def test_neo4j_needs_a_password(monkeypatch):
    monkeypatch.setattr(graph_kg, "NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setattr(graph_kg, "NEO4J_PASS", None)
    assert graph_kg._driver() is None


def test_agent_tool_errors_do_not_raise():
    agent = pytest.importorskip("copilot.agent")
    assert "error" in agent.run_tool("predict_rul", {"cycles": "nonsense"})
    assert "error" in agent.run_tool("predict_rul", {"wrong_arg": 1})
    assert "error" in agent.run_tool("predict_rul", None)
    assert "error" in agent.run_tool("no_such_tool", {})
    bad = agent.run_tool("failure_modes", {"sensor_name": "warp_core"})
    assert "valid_sensor_names" in bad


def test_agent_graph_runs_tool_loop_with_a_fake_llm(monkeypatch):
    """Drive the LangGraph loop with a scripted OpenAI-style client: the first
    turn calls a tool with broken arguments, the second answers."""
    pytest.importorskip("langgraph")
    import sys
    import types
    from copilot import agent

    def msg(content=None, calls=None):
        tcs = [types.SimpleNamespace(id=f"c{i}", function=types.SimpleNamespace(name=n, arguments=a))
               for i, (n, a) in enumerate(calls or [])]
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=content, tool_calls=tcs or None))])

    script = [msg(calls=[("predict_rul", "{not json"), ("failure_modes", '{"sensor_name": "sensor_3"}')]),
              msg(content="HPC degradation; inspect.")]
    seen = []

    class FakeOpenAI:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

        def _create(self, **kw):
            seen.append(kw["messages"])
            return script[len(seen) - 1]

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    out = agent.build_agent().ask("Engine 3 HPC temp rising?")
    assert out["answer"] == "HPC degradation; inspect."
    assert [t["tool"] for t in out["trace"]] == ["predict_rul", "failure_modes"]
    assert [t["ok"] for t in out["trace"]] == [False, True]
    tool_msgs = [m for m in seen[1] if m["role"] == "tool"]
    assert len(tool_msgs) == 2
