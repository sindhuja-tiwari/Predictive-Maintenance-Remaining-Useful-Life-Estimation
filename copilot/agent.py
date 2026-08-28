"""
Maintenance Copilot agent, built as a LangGraph state graph.

Graph shape:

    START -> agent -> (tools -> agent)* -> END

- `agent` node: calls the LLM with the tool schemas. The LLM decides whether to
  answer directly or call a tool.
- `tools` node: executes the requested tool(s) and appends results.
- Conditional edge routes back to `agent` while the LLM keeps requesting tools,
  otherwise to END.

Tools exposed to the model:
  predict_rul          -> the trained C-MAPSS model (copilot/rul_tool.py)
  search_manuals       -> RAG retrieval over maintenance PDFs (copilot/rag.py)
  failure_modes        -> Engine/Sensor/FailureMode graph (copilot/graph_kg.py)

LLM provider is selected by environment:
  ANTHROPIC_API_KEY -> Anthropic   |   OPENAI_API_KEY -> OpenAI
If neither is set, the agent runs in a degraded "retrieval-only" mode that still
answers from the manuals (RAG) without tool-calling reasoning.
"""
import os
import json
from typing import Annotated, TypedDict

from . import rul_tool, rag, graph_kg


# ----------------------------- tool implementations -----------------------------

def _tool_predict_rul(cycles, unit_id=1):
    return rul_tool.predict_rul(cycles, unit_id=unit_id)


def _tool_search_manuals(query, k=4):
    hits = rag.retrieve(query, k=k)
    return {"results": hits}


def _tool_failure_modes(sensor_name):
    return {"sensor": sensor_name, "modes": graph_kg.failure_modes_for_sensor(sensor_name)}


TOOLS = {
    "predict_rul": _tool_predict_rul,
    "search_manuals": _tool_search_manuals,
    "failure_modes": _tool_failure_modes,
}

TOOL_SCHEMAS = [
    rul_tool.RUL_TOOL_SCHEMA,
    {
        "name": "search_manuals",
        "description": "Search the maintenance manuals/knowledge base for guidance, "
                       "procedures, sensor meanings, or failure-mode explanations. "
                       "Use for any 'how do I', 'what does', 'what should I do' question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "k": {"type": "integer", "description": "Number of chunks (default 4)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "failure_modes",
        "description": "Given a sensor name, return the failure modes it indicates and "
                       "their mitigations, from the Engine->Sensor->FailureMode graph.",
        "parameters": {
            "type": "object",
            "properties": {
                "sensor_name": {"type": "string", "description": "e.g. 'vibration', 'fan_speed'."},
            },
            "required": ["sensor_name"],
        },
    },
]

SYSTEM_PROMPT = (
    "You are the Maintenance Copilot for turbofan-engine health. You help reliability "
    "engineers interpret Remaining Useful Life (RUL) predictions and decide on "
    "maintenance actions. Prefer calling tools over guessing: use predict_rul for any "
    "health/RUL estimate from sensor data, search_manuals for procedures and "
    "explanations, and failure_modes to map a sensor to likely failure causes. "
    "Ground your answers in tool results and cite the manual source when you use one. "
    "Be concise, practical, and safety-conscious; when a reading is CRITICAL, say so plainly."
)


# ----------------------------- LLM provider layer -----------------------------

def _provider():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def _anthropic_schemas():
    return [{"name": t["name"], "description": t["description"],
             "input_schema": t["parameters"]} for t in TOOL_SCHEMAS]


def _openai_schemas():
    return [{"type": "function", "function": t} for t in TOOL_SCHEMAS]


# ----------------------------- LangGraph state graph -----------------------------

class AgentState(TypedDict):
    messages: Annotated[list, "conversation messages"]
    answer: str
    trace: list  # tool calls made, for transparency


def _run_anthropic(messages):
    import anthropic
    client = anthropic.Anthropic()
    model = os.environ.get("COPILOT_MODEL", "claude-3-5-sonnet-20241022")
    return client, model


def build_agent():
    """Compile and return the LangGraph app. Falls back to retrieval-only if
    LangGraph or an LLM key is missing."""
    provider = _provider()
    try:
        from langgraph.graph import StateGraph, START, END
    except Exception as e:
        print(f"[agent] LangGraph unavailable ({e}); using retrieval-only fallback")
        return _RetrievalOnlyAgent()

    if provider is None:
        print("[agent] no LLM API key set; using retrieval-only fallback")
        return _RetrievalOnlyAgent()

    def agent_node(state: AgentState):
        msgs = state["messages"]
        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            model = os.environ.get("COPILOT_MODEL", "claude-3-5-sonnet-20241022")
            resp = client.messages.create(
                model=model, max_tokens=1024, system=SYSTEM_PROMPT,
                tools=_anthropic_schemas(), messages=msgs)
            tool_calls, text = [], ""
            for block in resp.content:
                if block.type == "text":
                    text += block.text
                elif block.type == "tool_use":
                    tool_calls.append({"id": block.id, "name": block.name, "args": block.input})
            state["messages"] = msgs + [{"role": "assistant", "content": resp.content}]
            state.setdefault("trace", [])
            if tool_calls:
                state["_pending"] = tool_calls
            else:
                state["answer"] = text
                state["_pending"] = []
        else:  # openai
            from openai import OpenAI
            client = OpenAI()
            model = os.environ.get("COPILOT_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(
                model=model, tools=_openai_schemas(),
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + msgs)
            m = resp.choices[0].message
            state["messages"] = msgs + [m.model_dump()]
            state.setdefault("trace", [])
            if m.tool_calls:
                state["_pending"] = [
                    {"id": tc.id, "name": tc.function.name,
                     "args": json.loads(tc.function.arguments or "{}")}
                    for tc in m.tool_calls]
            else:
                state["answer"] = m.content or ""
                state["_pending"] = []
        return state

    def tools_node(state: AgentState):
        pending = state.get("_pending", [])
        results = []
        for call in pending:
            fn = TOOLS.get(call["name"])
            out = fn(**call["args"]) if fn else {"error": f"unknown tool {call['name']}"}
            state["trace"].append({"tool": call["name"], "args": call["args"]})
            results.append((call, out))
        # append tool results in provider-specific format
        if provider == "anthropic":
            content = [{"type": "tool_result", "tool_use_id": c["id"],
                        "content": json.dumps(o)} for c, o in results]
            state["messages"].append({"role": "user", "content": content})
        else:
            for c, o in results:
                state["messages"].append({"role": "tool", "tool_call_id": c["id"],
                                          "content": json.dumps(o)})
        return state

    def route(state: AgentState):
        return "tools" if state.get("_pending") else END

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return _CompiledAgent(g.compile())


class _CompiledAgent:
    def __init__(self, app):
        self.app = app

    def ask(self, question):
        state = {"messages": [{"role": "user", "content": question}],
                 "answer": "", "trace": []}
        final = self.app.invoke(state)
        return {"answer": final.get("answer", ""), "trace": final.get("trace", [])}


class _RetrievalOnlyAgent:
    """Fallback with no LLM: answers by returning the most relevant manual chunks.
    Keeps the whole system runnable (and demoable) without any API key."""
    def ask(self, question):
        hits = rag.retrieve(question, k=3)
        if not hits:
            return {"answer": "No manual content indexed yet. Add PDFs to copilot/manuals/ "
                              "and rebuild the index.", "trace": []}
        answer = ("(retrieval-only mode — set ANTHROPIC_API_KEY or OPENAI_API_KEY for "
                  "full agentic reasoning)\n\nMost relevant guidance:\n\n")
        for h in hits:
            answer += f"• From {h['source']}: {h['text'][:400].strip()}...\n\n"
        return {"answer": answer.strip(),
                "trace": [{"tool": "search_manuals", "args": {"query": question}}]}


_AGENT = None


def get_agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()
    return _AGENT