import pytest

import app.agent.tool_loop as tool_loop_module
from app.agent.tool_loop import ToolLoop
from tests.test_tool_loop import (
    FakeClient,
    FakeGroundingLLM,
    FakeLLM,
    tool_result,
)


@pytest.mark.anyio
async def test_tool_trace_records_individual_duration(
    monkeypatch,
):
    client = FakeClient(
        results=[
            tool_result(
                content={
                    "files": ["README.md"]
                }
            )
        ],
        delay=0.02,
    )
    monkeypatch.setattr(
        tool_loop_module,
        "Client",
        lambda server: client,
    )
    llm = FakeLLM(
        responses=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "list_files",
                            "arguments": {"path": "."},
                        }
                    }
                ],
            },
            {"role": "assistant", "content": "Terminé."},
        ],
        generated=["Le projet contient README.md."],
    )
    agent = ToolLoop(
        llm=llm,
        grounding_llm=FakeGroundingLLM(),
        server=object(),
    )
    result = await agent.ask("Liste les fichiers.")
    assert result == (
        "Le projet contient README.md. [S1]"
    )
    assert len(agent.last_trace) == 1
    trace = agent.last_trace[0]
    assert trace["tool"] == "list_files"
    assert trace["status"] == "success"
    assert trace["arguments"] == {"path": "."}
    assert trace[
        "duration_seconds"
    ] >= 0.015
    assert (
        agent.last_stage_times[
            "tools"
        ]
        >= trace[
            "duration_seconds"
        ]
    )


@pytest.mark.anyio
async def test_duplicate_call_has_no_execution_duration(
    monkeypatch,
):
    client = FakeClient(
        results=[
            tool_result(
                content={
                    "files": ["README.md"]
                }
            )
        ]
    )
    monkeypatch.setattr(
        tool_loop_module,
        "Client",
        lambda server: client,
    )
    llm = FakeLLM(
        responses=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "list_files",
                            "arguments": {"path": "."},
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "list_files",
                            "arguments": {"path": "."},
                        }
                    }
                ],
            },
            {"role": "assistant", "content": "Terminé."},
        ],
        generated=["Le projet contient README.md."],
    )
    agent = ToolLoop(
        llm=llm,
        grounding_llm=FakeGroundingLLM(),
        server=object(),
    )
    await agent.ask("Liste les fichiers.")
    assert (
        agent.last_trace[0]["status"]
        == "success"
    )
    assert (
        agent.last_trace[1]["status"]
        == "duplicate"
    )
    assert (
        agent.last_trace[1][
            "duration_seconds"
        ]
        == 0.0
    )
