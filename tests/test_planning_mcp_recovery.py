import json

import pytest

from app.agent.tool_loop import ToolLoop
from app.mcp_server import mcp
from app.tools import project_tools
from tests.test_tool_loop import FakeGroundingLLM, FakeLLM


@pytest.mark.anyio
async def test_real_mcp_missing_path_diagnostic_reaches_model_before_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEAGENT_DEBUG", "1")
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    (tmp_path / "target.py").write_text("def marker():\n    return 42\n", encoding="utf-8")
    assert not (tmp_path / "absent.py").exists()
    goal = (
        'Localise marker avec un premier appel search_code(query="marker", path="absent.py"), '
        "puis lis son implémentation ; si cet appel échoue, corrige seulement son paramètre path."
    )
    plan = {"goal": goal, "steps": [
        {"id": "1", "objective": goal.split(", puis ")[0], "depends_on": [],
         "success": {"tool": "search_code", "kind": "contains", "value": "marker"}},
        {"id": "2", "objective": goal.split(", puis ")[1].rstrip("."), "depends_on": ["1"],
         "success": {"tool": "read_file", "kind": "nonempty", "from_step": "1"}},
    ]}

    def call(name, **arguments):
        return {"role": "assistant", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}

    candidate = (
        "La première recherche dans absent.py échoue car le chemin n'existe pas [S3].\n\n"
        "La recherche corrigée retrouve marker dans target.py [S1]. Son implémentation retourne 42 [S2]."
    )
    llm = FakeLLM([
        {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)},
        call("search_code", query="marker", path="absent.py"),
        call("search_code", query="marker", path="."),
        call("read_file", path="target.py", start_line=1, end_line=2),
    ], generated=[candidate])
    grounding = FakeGroundingLLM()
    agent = ToolLoop(llm=llm, grounding_llm=grounding, server=mcp)

    assert await agent.ask(goal) == candidate
    assert [trace["status"] for trace in agent.last_trace] == ["error", "success", "success"]
    assert [trace["tool"] for trace in agent.last_trace] == ["search_code", "search_code", "read_file"]
    assert [trace["arguments"]["path"] for trace in agent.last_trace] == ["absent.py", ".", "target.py"]
    assert "observation_id" not in agent.last_trace[0]
    assert [trace["observation_id"] for trace in agent.last_trace[1:]] == ["S1", "S2"]
    assert agent.execution.tool_calls == 3
    assert agent.last_execution["replans"] == 0
    assert agent.last_execution["stop_reason"] == "goal_achieved"
    assert [step["evidence"] for step in agent.last_execution["plan"]] == [["S1"], ["S2"]]

    retry_context = llm.calls[2]["messages"]
    error = json.loads(next(message["content"] for message in retry_context if message["role"] == "tool"))
    assert error["ok"] is False
    assert error["error"] == "tool_error"
    assert "Path does not exist: absent.py" in error["message"]
    assert error["message"] in agent.last_execution["errors"][0]
    assert "Path does not exist: absent.py" in retry_context[0]["content"]

    sources = agent.last_debug["sources"]
    assert len(sources) == 3
    assert sources[0]["observation_id"] == "S1"
    assert sources[0]["name"] == "search_code"
    assert '"path": "."' in sources[0]["content"]
    assert sources[1]["observation_id"] == "S2"
    assert sources[1]["name"] == "read_file"
    assert "return 42" in sources[1]["content"]
    assert sources[2]["type"] == "tool_error"
    assert "Path does not exist: absent.py" in sources[2]["content"]
    assert agent.last_debug["source_mapping"] == {"S1": "S1", "S2": "S2", "S3": None}
    assert agent.last_grounding == "passed"
    assert "[S3]" in grounding.calls[0]["prompt"]
    assert "Path does not exist: absent.py" in grounding.calls[0]["prompt"]
    assert "Tool: search_code" in grounding.calls[1]["prompt"]
    assert "return 42" in grounding.calls[2]["prompt"]
