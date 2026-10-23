"""Empty-session recall must not become generated personal history."""

import pytest

import app.agent.tool_loop as loop_module
from app.agent.tool_loop import ToolLoop
from app.memory.conversation import ConversationMemory
from app.utils.paths import project_memory_dir
from tests.test_tool_loop import FakeClient, FakeGroundingLLM, FakeLLM, tool_result


def assert_missing_history(answer):
    text = answer.casefold()
    assert "session" in text
    assert any(marker in text for marker in ("aucun", "absen", "ne retrouve", "pas d'historique"))


@pytest.mark.anyio
@pytest.mark.parametrize("question", [
    "Quel code et quelle contrainte ai-je indiqués en dernier dans cette session ?",
    "Quelle décision avais-je prise dans cette conversation ?",
    "Quel nom avons-nous choisi ?",
    "Rappelle ma priorité dans cette session.",
    "What code did I choose for the project?",
    "Which constraint have I mentioned in this conversation?",
])
async def test_empty_history_recall_abstains_before_models_and_mcp(question, monkeypatch):
    def unexpected_client(server):
        pytest.fail("Pure recall of an empty session must not open MCP.")

    monkeypatch.setattr(loop_module, "Client", unexpected_client)
    llm, grounding = FakeLLM([]), FakeGroundingLLM()
    memory = ConversationMemory()
    agent = ToolLoop(llm=llm, grounding_llm=grounding, server=object(), memory=memory)

    assert_missing_history(await agent.ask(question))

    assert llm.calls == llm.generate_calls == grounding.calls == []
    assert agent.last_trace == []
    assert agent.last_token_stats["total"] == 0
    assert agent.last_execution["mode"] == "M6_DIRECT"
    assert agent.last_execution["plan"] == []
    assert memory.total_turns == 1  # The abstention remains an ordinary archived exchange.


@pytest.mark.anyio
@pytest.mark.parametrize("isolation", ["session", "project"])
async def test_empty_recall_uses_selected_persisted_namespace(tmp_path, monkeypatch, isolation):
    original = project_memory_dir(tmp_path / "a" / "project", tmp_path / "memory")
    known = ConversationMemory(original, "selected")
    known.record("Le nom retenu est VEGA ; contrainte : lecture seule.", "Enregistré.")
    selected = (ConversationMemory(original, "other") if isolation == "session" else
                ConversationMemory(project_memory_dir(tmp_path / "b" / "project", tmp_path / "memory"), "selected"))
    assert selected.total_turns == 0

    def unexpected_client(server):
        pytest.fail("An unrelated archive must not bypass the empty-session guard.")

    monkeypatch.setattr(loop_module, "Client", unexpected_client)
    llm, grounding = FakeLLM([]), FakeGroundingLLM()
    agent = ToolLoop(llm=llm, grounding_llm=grounding, server=object(), memory=selected)
    answer = await agent.ask("Quel nom ai-je choisi dans cette session ?")

    assert_missing_history(answer)
    assert "VEGA" not in answer
    assert llm.calls == grounding.calls == []
    restored = ConversationMemory(original, "selected")
    assert restored.total_turns == 1
    assert "VEGA" in restored.recall("nom")


@pytest.mark.anyio
async def test_nonempty_recall_keeps_latest_correction_after_restart(tmp_path, monkeypatch):
    memory = ConversationMemory(tmp_path, "selected")
    memory.record("Le nom de mission est VEGA.", "Enregistré.")
    memory.record("Correction : le nom de mission est LYRA.", "Corrigé.")
    restored = ConversationMemory(tmp_path, "selected")
    client = FakeClient()
    monkeypatch.setattr(loop_module, "Client", lambda server: client)
    llm, grounding = FakeLLM([{"role": "assistant", "content": "LYRA"}]), FakeGroundingLLM()
    agent = ToolLoop(llm=llm, grounding_llm=grounding, server=object(), memory=restored)

    assert await agent.ask("Quel nom de mission ai-je indiqué en dernier ?") == "LYRA"

    assert len(llm.calls) == 1
    assert client.calls == grounding.calls == []
    assert "Correction : le nom de mission est LYRA." in str(llm.calls[0]["messages"])
    assert ConversationMemory(tmp_path, "selected").total_turns == 3


@pytest.mark.anyio
@pytest.mark.parametrize("question", [
    "Bonjour.",
    "Dans cette session, retiens le nom VEGA et la contrainte lecture seule.",
    "Correction : le nom est LYRA ; conserve lecture seule.",
    "Qu'est-ce qu'une session HTTP ?",
    "J'ai un problème Python. Comment le corriger ?",
    "Quel nom de mission pourrais-je choisir ?",
    "J’ai choisi VEGA comme nom de mission. Retiens ce nom pour cette session.",
    "J'ai choisi VEGA pour cette session.",
    "Rappelle-toi ma priorité : lecture seule.",
    "Ai-je besoin d'installer Python ?",
])
async def test_empty_history_does_not_block_other_conversation(question, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(loop_module, "Client", lambda server: client)
    llm, grounding = FakeLLM([{"role": "assistant", "content": "OK"}]), FakeGroundingLLM()
    agent = ToolLoop(llm=llm, grounding_llm=grounding, server=object())

    assert await agent.ask(question) == "OK"

    assert len(llm.calls) == 1
    assert client.calls == grounding.calls == []


@pytest.mark.anyio
async def test_empty_history_mixed_recall_preserves_real_inspection(monkeypatch):
    client = FakeClient([tool_result(content={"path": "current.py", "start_line": 1,
                                            "end_line": 1, "content": "VALUE = 42"})])
    monkeypatch.setattr(loop_module, "Client", lambda server: client)
    llm = FakeLLM([
        {"role": "assistant", "tool_calls": [{"function": {
            "name": "read_file", "arguments": {"path": "current.py"},
        }}]},
        {"role": "assistant", "content": "Terminé."},
    ], generated=["VALUE vaut 42 [S1]."])
    grounding = FakeGroundingLLM()
    agent = ToolLoop(llm=llm, grounding_llm=grounding, server=object())

    answer = await agent.ask("Rappelle ma priorité dans cette session et lis current.py.")

    assert answer == "VALUE vaut 42 [S1]."
    assert client.calls == [("read_file", {"path": "current.py"})]
    assert len(grounding.calls) == 1
    assert agent.last_grounding == "passed"
