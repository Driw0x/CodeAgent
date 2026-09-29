import asyncio
from types import SimpleNamespace

import pytest

import app.agent.tool_loop as tool_loop_module
from app.agent.tool_loop import ToolLoop
from app.mcp_server import mcp
from app.tools import project_tools


class FakeLLM:
    def __init__(self, responses, generated=None):
        self.responses = list(responses)
        self.generated = list(generated or [])
        self.calls = []
        self.generate_calls = []
        self.last_stats = None
        self.usage_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def reset_usage_stats(self):
        self.usage_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def chat(self, messages, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        self.usage_stats["prompt_tokens"] += 10
        self.usage_stats["completion_tokens"] += 2
        self.usage_stats["total_tokens"] += 12
        return self.responses.pop(0)

    def generate(self, prompt, system_prompt=None):
        self.generate_calls.append({"prompt": prompt, "system_prompt": system_prompt})
        self.usage_stats["prompt_tokens"] += 20
        self.usage_stats["completion_tokens"] += 3
        self.usage_stats["total_tokens"] += 23
        return self.generated.pop(0)


class FakeGroundingLLM:
    def __init__(self, verdict="SUPPORTED"):
        self.verdict = verdict
        self.calls = []
        self.last_stats = None
        self.usage_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def reset_usage_stats(self):
        self.usage_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def generate(self, prompt, system_prompt=None):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        self.usage_stats["prompt_tokens"] += 30
        self.usage_stats["completion_tokens"] += 1
        self.usage_stats["total_tokens"] += 31
        return self.verdict


class FakeClient:
    def __init__(self, results=None, delay=0):
        self.results = list(results or [])
        self.delay = delay
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="list_files",
                    description="List project files.",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                ),
                SimpleNamespace(
                    name="read_file",
                    description="Read a project file.",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                ),
            ]
        )

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.results.pop(0)


def tool_result(is_error=False, content=None):
    return SimpleNamespace(
        is_error=is_error,
        content=[],
        structured_content=content,
    )


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_tool_loop_answers_without_tool():
    llm = FakeLLM(
        [
            {"role": "assistant", "content": "OK"}
        ]
    )
    grounding_llm = FakeGroundingLLM()
    agent = ToolLoop(
        llm=llm,
        grounding_llm=grounding_llm,
        server=mcp,
    )
    result = await agent.ask("Réponds uniquement par OK.")
    assert result == "OK"
    assert len(llm.calls) == 1
    assert llm.generate_calls == []
    assert grounding_llm.calls == []
    assert agent.last_trace == []
    assert agent.last_grounding == "not_used"


@pytest.mark.anyio
async def test_tool_loop_adds_single_source_citation_and_grounds(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        project_tools,
        "PROJECT_ROOT",
        tmp_path,
    )
    rag_dir = tmp_path / "app" / "rag"
    rag_dir.mkdir(parents=True)
    (rag_dir / "pipeline.py").write_text("", encoding="utf-8")
    (rag_dir / "grounding.py").write_text("", encoding="utf-8")
    llm = FakeLLM(
        responses=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "list_files",
                            "arguments": {"path": "app/rag"},
                        }
                    }
                ],
            },
            {"role": "assistant", "content": "Tool terminé."},
        ],
        generated=[
            (
                "Le dossier `app/rag` contient `pipeline.py` "
                "et `grounding.py`."
            )
        ],
    )
    grounding_llm = FakeGroundingLLM()
    agent = ToolLoop(
        llm=llm,
        grounding_llm=grounding_llm,
        server=mcp,
    )
    result = await agent.ask("Liste les fichiers du dossier app/rag.")
    assert result.endswith("[S1]")
    assert agent.last_grounding == "passed"
    assert len(llm.generate_calls) == 1
    assert "[S1]" in llm.generate_calls[0]["prompt"]
    assert "list_files" in llm.generate_calls[0]["prompt"]
    assert "pipeline.py" in llm.generate_calls[0]["prompt"]
    assert len(grounding_llm.calls) == 1


@pytest.mark.anyio
async def test_tool_loop_generates_grounded_final_answer_from_read_file(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        project_tools,
        "PROJECT_ROOT",
        tmp_path,
    )
    (tmp_path / "README.md").write_text("# CodeAgent\nMCP works.", encoding="utf-8")
    llm = FakeLLM(
        responses=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "README.md"},
                        }
                    }
                ],
            },
            {"role": "assistant", "content": "Terminé."},
        ],
        generated=["Le README indique que MCP fonctionne."],
    )
    grounding_llm = FakeGroundingLLM()
    agent = ToolLoop(
        llm=llm,
        grounding_llm=grounding_llm,
        server=mcp,
    )
    result = await agent.ask("Lis README.md.")
    assert result == (
        "Le README indique que MCP fonctionne. [S1]"
    )
    assert agent.last_grounding == "passed"
    assert len(grounding_llm.calls) == 1


@pytest.mark.anyio
async def test_tool_loop_rejects_invalid_citation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        project_tools,
        "PROJECT_ROOT",
        tmp_path,
    )
    (tmp_path / "README.md").write_text("# CodeAgent", encoding="utf-8")
    llm = FakeLLM(
        responses=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "README.md"},
                        }
                    }
                ],
            },
            {"role": "assistant", "content": "Terminé."},
        ],
        generated=["Le README contient CodeAgent. [S2]"],
    )
    grounding_llm = FakeGroundingLLM()
    agent = ToolLoop(
        llm=llm,
        grounding_llm=grounding_llm,
        server=mcp,
    )
    result = await agent.ask("Lis README.md.")
    assert result == (
        "La réponse générée contient des citations "
        "invalides ou manquantes."
    )
    assert agent.last_grounding == "citation_error"
    assert grounding_llm.calls == []


@pytest.mark.anyio
async def test_tool_loop_rejects_unsupported_claim(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        project_tools,
        "PROJECT_ROOT",
        tmp_path,
    )
    (tmp_path / "README.md").write_text("# CodeAgent", encoding="utf-8")
    llm = FakeLLM(
        responses=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "README.md"},
                        }
                    }
                ],
            },
            {"role": "assistant", "content": "Terminé."},
        ],
        generated=["CodeAgent utilise Kubernetes en production."],
    )
    grounding_llm = FakeGroundingLLM(verdict="UNSUPPORTED")
    agent = ToolLoop(
        llm=llm,
        grounding_llm=grounding_llm,
        server=mcp,
    )
    result = await agent.ask("Lis README.md.")
    assert result == (
        "La réponse générée contient des affirmations "
        "non supportées par les sources."
    )
    assert agent.last_grounding == "unsupported"
    assert len(grounding_llm.calls) == 1


@pytest.mark.anyio
async def test_tool_loop_detects_duplicate_call(
    monkeypatch,
):
    client = FakeClient(
        results=[
            tool_result(
                content={"files": ["README.md"], "truncated": False}
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
    result = await agent.ask("Liste les fichiers.")
    assert result == (
        "Le projet contient README.md. [S1]"
    )
    assert client.calls == [
        (
            "list_files",
            {"path": "."},
        )
    ]
    assert agent.last_trace[1]["status"] == "duplicate"


@pytest.mark.anyio
async def test_tool_loop_recovers_after_tool_error(
    monkeypatch,
):
    client = FakeClient(
        results=[
            tool_result(
                is_error=True,
                content={"message": "File does not exist."},
            ),
            tool_result(
                content={"path": "README.md", "content": "# CodeAgent"}
            ),
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
                            "name": "read_file",
                            "arguments": {"path": "missing.md"},
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
                            "name": "read_file",
                            "arguments": {"path": "README.md"},
                        }
                    }
                ],
            },
            {"role": "assistant", "content": "Terminé."},
        ],
        generated=["Le README contient le titre CodeAgent."],
    )
    agent = ToolLoop(
        llm=llm,
        grounding_llm=FakeGroundingLLM(),
        server=object(),
    )
    result = await agent.ask("Lis le README.")
    assert result == (
        "Le README contient le titre CodeAgent. [S1]"
    )
    assert client.calls == [
        (
            "read_file",
            {"path": "missing.md"},
        ),
        (
            "read_file",
            {"path": "README.md"},
        ),
    ]
    assert agent.last_trace[0]["status"] == "error"
    assert agent.last_trace[1]["status"] == "success"


@pytest.mark.anyio
async def test_tool_loop_limits_retries(
    monkeypatch,
):
    client = FakeClient(
        results=[
            tool_result(
                is_error=True,
                content={"message": "Missing file."},
            ),
            tool_result(
                is_error=True,
                content={"message": "Missing file."},
            ),
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
                            "name": "read_file",
                            "arguments": {"path": "missing1.md"},
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
                            "name": "read_file",
                            "arguments": {"path": "missing2.md"},
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
                            "name": "read_file",
                            "arguments": {"path": "missing3.md"},
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": (
                    "Impossible de lire le fichier."
                ),
            },
        ]
    )
    agent = ToolLoop(
        llm=llm,
        grounding_llm=FakeGroundingLLM(),
        server=object(),
        max_retries=1,
    )
    result = await agent.ask("Lis un fichier.")
    assert result == (
        "Impossible de lire le fichier."
    )
    assert len(client.calls) == 2
    assert agent.last_trace[-1]["status"] == "retry_limit"


@pytest.mark.anyio
async def test_tool_loop_handles_timeout(
    monkeypatch,
):
    client = FakeClient(
        results=[
            tool_result(
                content={"files": ["README.md"]}
            )
        ],
        delay=0.05,
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
            {"role": "assistant", "content": "Le tool a expiré."},
        ]
    )
    agent = ToolLoop(
        llm=llm,
        grounding_llm=FakeGroundingLLM(),
        server=object(),
        tool_timeout=0.01,
    )
    result = await agent.ask("Liste les fichiers.")
    assert result == "Le tool a expiré."
    assert agent.last_trace[-1]["status"] == "timeout"


@pytest.mark.anyio
async def test_tool_loop_generates_final_answer_at_max_steps_when_sources_exist(
    monkeypatch,
):
    client = FakeClient(
        results=[
            tool_result(
                content={"files": ["README.md"]}
            ),
            tool_result(
                content={"files": ["README.md"]}
            ),
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
                            "arguments": {"path": "app"},
                        }
                    }
                ],
            },
        ],
        generated=[
            (
                "Le premier listing contient `README.md` [S1]. "
                "Le second listing contient également `README.md` [S2]."
            )
        ],
    )
    agent = ToolLoop(
        llm=llm,
        grounding_llm=FakeGroundingLLM(),
        server=object(),
        max_steps=2,
    )
    result = await agent.ask("Liste plusieurs dossiers.")
    assert result == (
        "Le premier listing contient `README.md` [S1]. "
        "Le second listing contient également `README.md` [S2]."
    )
    assert agent.last_steps == 2
    assert len(agent.last_trace) == 2
    assert len(llm.generate_calls) == 1
    assert agent.last_grounding == "passed"


@pytest.mark.anyio
async def test_tool_loop_raises_at_max_steps_without_sources(
    monkeypatch,
):
    client = FakeClient(
        results=[
            tool_result(
                is_error=True,
                content={"message": "Missing file."},
            ),
            tool_result(
                is_error=True,
                content={"message": "Missing file."},
            ),
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
                            "name": "read_file",
                            "arguments": {"path": "missing1.py"},
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
                            "name": "read_file",
                            "arguments": {"path": "missing2.py"},
                        }
                    }
                ],
            },
        ]
    )
    agent = ToolLoop(
        llm=llm,
        grounding_llm=FakeGroundingLLM(),
        server=object(),
        max_steps=2,
    )
    with pytest.raises(RuntimeError, match="Maximum number of tool steps reached"):
        await agent.ask("Lis un fichier existant.")


@pytest.mark.anyio
async def test_tool_loop_accumulates_token_usage(
    monkeypatch,
):
    client = FakeClient(
        results=[
            tool_result(
                content={"path": "README.md", "content": "# CodeAgent"}
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
                            "name": "read_file",
                            "arguments": {"path": "README.md"},
                        }
                    }
                ],
            },
            {"role": "assistant", "content": "Terminé."},
        ],
        generated=["Le README contient CodeAgent."],
    )
    grounding_llm = FakeGroundingLLM()
    agent = ToolLoop(
        llm=llm,
        grounding_llm=grounding_llm,
        server=object(),
    )
    await agent.ask("Lis README.md.")
    assert agent.last_token_stats == {
        "devstral_prompt": 40,
        "devstral_completion": 7,
        "grounding_prompt": 30,
        "grounding_completion": 1,
        "total": 78,
    }


@pytest.mark.anyio
async def test_tool_loop_resets_token_usage_between_questions():
    llm = FakeLLM(
        responses=[
            {"role": "assistant", "content": "OK"},
            {"role": "assistant", "content": "OK"},
        ]
    )
    agent = ToolLoop(
        llm=llm,
        grounding_llm=FakeGroundingLLM(),
        server=mcp,
    )
    await agent.ask("Réponds uniquement par OK.")
    first = dict(agent.last_token_stats)
    await agent.ask("Réponds uniquement par OK.")
    second = dict(agent.last_token_stats)
    assert first["devstral_prompt"] == 10
    assert first["devstral_completion"] == 2
    assert first["total"] == 12
    assert second == first
