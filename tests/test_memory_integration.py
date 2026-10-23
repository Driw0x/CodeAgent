import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from markdown_it import MarkdownIt

import app.agent.tool_loop as loop_module
from app.agent.tool_loop import ToolLoop, normalize_single_source_citations
from app.llm import LocalLLM
from app.memory.conversation import ConversationMemory
from app.rag import RAGPipeline
from app.rag.grounding import extract_claims
from tests.test_rag_grounding import source
from tests.test_rag_pipeline import make_chunk
from tests.test_tool_loop import FakeClient, FakeGroundingLLM, FakeLLM, tool_result


def test_rag_followup_restores_memory_and_keeps_grounding(tmp_path):
    memory = ConversationMemory(tmp_path, "rag")
    memory.record("Décision : utiliser read_file", "Choix enregistré")
    llm = MagicMock()
    llm.generate.side_effect = ["read_file retourne du texte [S1].", "SUPPORTED"]
    retriever = MagicMock(return_value=[make_chunk()])
    pipeline = RAGPipeline(retriever, llm, memory=ConversationMemory(tmp_path, "rag"))
    answer = pipeline.ask("Que fait ce choix ?")
    assert "read_file" in retriever.call_args.args[0]
    assert "Décision : utiliser read_file" in llm.generate.call_args_list[0].kwargs["prompt"]
    assert "HISTORIQUE CONVERSATIONNEL" not in llm.generate.call_args_list[1].kwargs["prompt"]
    assert "Sources:" in answer
    assert ConversationMemory(tmp_path, "rag").total_turns == 2


def test_rag_large_sources_keep_supported_source_and_contiguous_ids():
    large = {**make_chunk(), "file": "app/large.py", "content": "x" * 20_000}
    llm = MagicMock()
    llm.context_tokens = 5500
    llm.num_predict = 300
    llm.generate.side_effect = ["read_file retourne du texte [S1].", "SUPPORTED"]
    pipeline = RAGPipeline(lambda q, k: [large, make_chunk()], llm)
    assert "Sources:" in pipeline.ask("Que fait read_file ?")
    prompt = llm.generate.call_args_list[0].kwargs["prompt"]
    assert "app/large.py" not in prompt
    assert "[S2]" not in prompt
    assert pipeline.last_context_stats["estimated_tokens"] <= pipeline.budget.input_limit


@pytest.mark.anyio
async def test_tool_loop_multiturn_restart_and_exact_no_tool_recall(tmp_path):
    class RecallLLM(FakeLLM):
        def chat(self, messages, tools=None):
            self.calls.append({"messages": list(messages), "tools": tools})
            content = "\n".join(message.get("content", "") for message in messages[:-1])
            return {"role": "assistant", "content": "ORION" if "ORION" in content else "inconnu"}
    first = ToolLoop(llm=FakeLLM([{"role": "assistant", "content": "OK"}]), grounding_llm=FakeGroundingLLM(),
                     server=loop_module.build_server_parameters(), memory=ConversationMemory(tmp_path, "agent"))
    # Real in-memory MCP transport, no Ollama or subprocess.
    from app.mcp_server import mcp
    first.server = mcp
    await first.ask("Décision : nom de mission ORION. Réponds OK.")
    llm = RecallLLM([])
    second = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=mcp,
                      memory=ConversationMemory(tmp_path, "agent"))
    assert await second.ask("Quel nom de mission ?") == "ORION"
    assert second.last_trace == []
    assert len(llm.calls) == 1
    assert ConversationMemory(tmp_path, "agent").total_turns == 2


@pytest.mark.anyio
async def test_large_tool_observation_preserves_original_and_grounding_subset(tmp_path, monkeypatch):
    raw = {"path": "app/target.py", "start_line": 1, "end_line": 600,
           "content": "ordinary = 0\n" * 599 + "ORION = 42"}
    client = FakeClient([tool_result(content=raw)])
    monkeypatch.setattr(loop_module, "Client", lambda server: client)
    llm = FakeLLM([
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "app/target.py"}}}]},
        {"role": "assistant", "content": "Terminé"},
    ], generated=["ORION vaut 42 [S1]."])
    llm.context_tokens = 7000
    llm.num_predict = 300
    grounding = FakeGroundingLLM()
    memory = ConversationMemory(tmp_path, "tool")
    agent = ToolLoop(llm=llm, grounding_llm=grounding, server=object(), memory=memory)
    assert await agent.ask("Lis ORION dans app/target.py") == "ORION vaut 42 [S1]."
    second_messages = llm.calls[1]["messages"]
    observation = next(message for message in second_messages if message["role"] == "tool")
    json.loads(observation["content"])
    assert "ORION = 42" in observation["content"]
    final_prompt = llm.generate_calls[0]["prompt"]
    assert "ORION = 42" in final_prompt
    assert "600" in final_prompt
    assert "ORION = 42" in grounding.calls[0]["prompt"]
    archived = json.loads((tmp_path / "history.jsonl").read_text(encoding="utf-8"))
    original = json.loads(archived["observations"][1]["content"])["result"]
    assert json.loads(original) == raw
    assert len(final_prompt) < len(original)


@pytest.mark.anyio
async def test_old_citation_cannot_be_reused_as_current_source(monkeypatch):
    monkeypatch.setattr(loop_module, "Client", lambda server: FakeClient())
    memory = ConversationMemory()
    memory.record("Ancien fichier", "Le fichier existe [S1].")
    llm = FakeLLM([{"role": "assistant", "content": "Le fichier existe [S1]."}])
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=object(), memory=memory)
    assert "citations invalides" in await agent.ask("Existe-t-il toujours ?")
    assert agent.last_grounding == "citation_error"
    assert "[S1]" not in json.dumps(llm.calls[0]["messages"][1:])


@pytest.mark.anyio
async def test_tool_loop_failure_is_restored_as_unresolved(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_module, "Client", lambda server: FakeClient())
    llm = FakeLLM([{"role": "assistant", "content": ""}])
    memory = ConversationMemory(tmp_path, "error")
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=object(), memory=memory)
    with pytest.raises(RuntimeError, match="neither"):
        await agent.ask("Question ORION")
    restored = ConversationMemory(tmp_path, "error")
    assert "Erreur non résolue" in restored.recall("ORION erreur")


@pytest.mark.parametrize("method", ["generate", "chat"])
def test_local_llm_budget_blocks_network_and_sends_num_ctx(method, monkeypatch):
    llm = LocalLLM(context_tokens=2000, num_predict=300)
    request = MagicMock(return_value={"response": "OK", "message": {"role": "assistant", "content": "OK"}})
    monkeypatch.setattr(llm, "_request", request)
    if method == "generate":
        llm.generate("court", "System")
    else:
        llm.chat([{"role": "user", "content": "court"}])
    assert request.call_args.args[1]["options"]["num_ctx"] == 2000
    request.reset_mock()
    with pytest.raises(RuntimeError, match="Context budget exceeded"):
        if method == "generate":
            llm.generate("x" * 3000)
        else:
            llm.chat([{"role": "user", "content": "x" * 3000}])
    request.assert_not_called()


def test_rag_failure_is_restored_as_unresolved(tmp_path):
    llm = MagicMock()
    llm.generate.side_effect = RuntimeError("Ollama unavailable")
    pipeline = RAGPipeline(lambda q, k: [make_chunk()], llm, memory=ConversationMemory(tmp_path))
    with pytest.raises(RuntimeError, match="Ollama unavailable"):
        pipeline.ask("Question read_file")
    assert "Erreur non résolue" in ConversationMemory(tmp_path).recall("Ollama erreur")


def test_explicit_rag_question_does_not_retrieve_unrelated_history():
    memory = ConversationMemory()
    memory.record("Décision de nom ORION", "OK")
    assert memory.retrieval_question("Où est définie read_file ?") == "Où est définie read_file ?"


@pytest.mark.anyio
async def test_stdio_mcp_uses_selected_project_and_schema(tmp_path, monkeypatch):
    from mcp import Client
    from app.agent.tool_loop import build_server_parameters
    (tmp_path / "selected.py").write_text("ORION = 42\n", encoding="utf-8")
    monkeypatch.setenv("CODEAGENT_PROJECT_PATH", str(tmp_path))
    async with Client(build_server_parameters()) as client:
        result = await client.call_tool("read_file", {"path": "selected.py"})
        invalid = await client.call_tool("read_file", {"path": "selected.py", "start_line": "invalid"})
    assert result.is_error is False
    content = loop_module.tool_result_content(result)
    assert "ORION = 42" in content
    assert invalid.is_error is True


@pytest.mark.anyio
async def test_nested_taskgroup_failure_preserves_leaf(tmp_path, monkeypatch):
    error = ExceptionGroup("unhandled errors in a TaskGroup", [
        ExceptionGroup("nested", [AttributeError("diagnostic leaf")]),
        RuntimeError("second leaf"),
    ])
    monkeypatch.setenv("CODEAGENT_DEBUG", "1")
    monkeypatch.setattr(loop_module, "Client", lambda server: FakeClient())
    llm = FakeLLM([])
    def fail(**kwargs):
        raise error
    llm.chat = fail
    memory = ConversationMemory(tmp_path, "failure")
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=object(), memory=memory)
    with pytest.raises(ExceptionGroup) as caught:
        await agent.ask("Question diagnostic")
    assert caught.value is error
    restored = ConversationMemory(tmp_path, "failure")
    text = restored.recall("diagnostic erreur", 2500)
    assert "AttributeError: diagnostic leaf" in text
    assert "RuntimeError: second leaf" in text
    assert restored.total_turns == 1


@pytest.mark.parametrize("debug", [False, True])
def test_agent_cli_prints_leaf_and_optional_full_traceback(tmp_path, monkeypatch, capsys, debug):
    import app.main as main_module
    monkeypatch.setenv("CODEAGENT_DEBUG", "1" if debug else "0")
    monkeypatch.setattr(loop_module, "Client", lambda server: FakeClient())
    llm = FakeLLM([])
    def fail(**kwargs):
        raise ExceptionGroup("TaskGroup", [AttributeError("traceback leaf")])
    llm.chat = fail
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=object(), memory=ConversationMemory(tmp_path))
    agent.llm.model = "fixture"
    agent.grounding_llm.model = "fixture"
    monkeypatch.setattr(main_module, "ConversationMemory", lambda *args, **kwargs: agent.memory)
    monkeypatch.setattr(main_module, "ToolLoop", lambda **kwargs: agent)
    monkeypatch.setattr(main_module, "preload_model", lambda *args: None)
    inputs = iter(["Question", ""])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    main_module.run_agent()
    output = capsys.readouterr()
    assert "AttributeError: traceback leaf" in output.out
    if debug:
        assert "Traceback" in output.err
        assert "AttributeError: traceback leaf" in output.err
        assert "in fail" in output.err
    else:
        assert output.err == ""


@pytest.mark.anyio
async def test_mcp_read_beyond_eof_reports_bounds_and_recovers(tmp_path, monkeypatch):
    from app.mcp_server import mcp
    from app.tools import project_tools
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    (tmp_path / "short.py").write_text("ORION = 42\n" * 226, encoding="utf-8")
    llm = FakeLLM([
        {"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "short.py", "start_line": 227}}}]},
        {"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "short.py", "start_line": 226, "end_line": 277}}}]},
        {"role": "assistant", "content": "Terminé"},
    ], generated=["ORION vaut 42 [S1]."])
    memory = ConversationMemory(tmp_path / "memory")
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=mcp, memory=memory)
    assert await agent.ask("Lis ORION dans short.py") == "ORION vaut 42 [S1]."
    error = json.loads(llm.calls[1]["messages"][-1]["content"])
    assert "start_line exceeds file length (226 lines)" in error["message"]
    assert [trace["status"] for trace in agent.last_trace] == ["error", "success"]
    assert agent.last_trace[0]["arguments"]["end_line"] == 277
    assert '"end_line": 226' in agent.grounding_llm.calls[0]["prompt"]


@pytest.mark.anyio
async def test_final_answer_preserves_recovered_range_error(tmp_path, monkeypatch):
    from app.mcp_server import mcp
    from app.tools import project_tools
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    (tmp_path / "short.py").write_text("ORION = 42\n" * 226, encoding="utf-8")
    llm = FakeLLM([
        {"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "short.py", "start_line": 227, "end_line": 277}}}]},
        {"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "short.py", "start_line": 200, "end_line": 226}}}]},
        {"role": "assistant", "content": "Terminé"},
    ])
    original_generate = llm.generate
    def generate(prompt, system_prompt=None):
        # Reproduce the abstention when the final generator loses the bounds error.
        if "start_line exceeds file length (226 lines)" in prompt:
            llm.generated.append("La plage 227–277 dépasse les 226 lignes du fichier [S2].\n\nLa lecture corrigée couvre les lignes 200–226, où ORION vaut 42 [S1].")
        else:
            llm.generated.append("Je ne peux pas déterminer cela à partir des sources disponibles.")
        return original_generate(prompt, system_prompt)
    llm.generate = generate
    grounding = FakeGroundingLLM()
    agent = ToolLoop(llm=llm, grounding_llm=grounding, server=mcp)
    result = await agent.ask("Lis short.py des lignes 227 à 277. Si la plage est invalide, corrige-la.")
    assert "lecture corrigée" in result
    assert agent.last_grounding == "passed"
    assert [trace["status"] for trace in agent.last_trace] == ["error", "success"]
    prompt = llm.generate_calls[0]["prompt"]
    assert prompt.index("[S2]") < prompt.index("[S1]")
    assert "start_line exceeds file length (226 lines)" in grounding.calls[0]["prompt"]
    assert '"start_line": 227' in grounding.calls[0]["prompt"]
    assert "ORION = 42" in grounding.calls[1]["prompt"]
    assert "[S2]" in grounding.calls[0]["prompt"]
    assert "[S1]" in grounding.calls[1]["prompt"]


@pytest.mark.anyio
async def test_recovered_error_keeps_large_source_with_memory(tmp_path, monkeypatch):
    from scripts.benchmark_memory import FixtureLLM
    from app.mcp_server import mcp
    from app.tools import project_tools
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    (tmp_path / "app").mkdir()
    (tmp_path / "app/target.py").write_text("ordinary = 0\n" * 599 + "ORION = 42\n", encoding="utf-8")
    llm = FixtureLLM([])
    memory = ConversationMemory(max_tokens=1800)
    memory.record("Référence : app/target.py:600, ORION.", "Détail historique " * 80)
    agent = ToolLoop(llm=llm, grounding_llm=llm, server=mcp, memory=memory)
    assert await agent.ask("Lis ORION dans app/target.py et récupère une erreur de lecture.") == "ORION vaut 42 [S1]."
    assert agent.last_grounding == "passed"
    assert max(llm.sizes) <= agent.budget.input_limit


@pytest.mark.parametrize("verdict", ["SUPPORTED", "UNSUPPORTED"])
@pytest.mark.parametrize("answer", [
    "ORION vaut 42 [S1]. Je ne peux pas déterminer le reste à partir des sources disponibles.",
    "ORION vaut 42.\n\nJe ne peux pas déterminer le reste à partir des sources disponibles [S1].",
])
def test_partial_abstention_still_verifies_recovered_facts(verdict, answer):
    grounding = FakeGroundingLLM(verdict)
    agent = ToolLoop(llm=FakeLLM([]), grounding_llm=grounding, server=object())
    result = agent._finalize_answer(answer, [make_chunk()])
    assert len(grounding.calls) == 1
    assert agent.last_grounding == ("passed" if verdict == "SUPPORTED" else "unsupported")
    if verdict == "SUPPORTED":
        assert "ORION vaut 42" in result
        assert "[S1]" in result.split("Je ne peux pas")[0]
    else:
        assert result == "La réponse générée contient des affirmations non supportées par les sources."


@pytest.mark.anyio
async def test_unrecovered_tool_error_preserves_abstention(monkeypatch):
    client = FakeClient([tool_result(is_error=True, content={"message": "File does not exist."})])
    monkeypatch.setattr(loop_module, "Client", lambda server: client)
    abstention = "Je ne peux pas le déterminer à partir des sources disponibles."
    llm = FakeLLM([
        {"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "missing.py"}}}]},
        {"role": "assistant", "content": abstention},
    ])
    grounding = FakeGroundingLLM()
    agent = ToolLoop(llm=llm, grounding_llm=grounding, server=object())
    assert await agent.ask("Lis missing.py") == abstention
    assert llm.generate_calls == []
    assert grounding.calls == []


@pytest.mark.anyio
@pytest.mark.parametrize("debug", [False, True])
async def test_grounding_debug_is_opt_in_and_preserves_strict_rejection(tmp_path, monkeypatch, debug):
    monkeypatch.setenv("CODEAGENT_DEBUG", "1" if debug else "0")
    client = FakeClient([tool_result(content={"path": "target.py", "content": "ORION = 42", "start_line": 1, "end_line": 1})])
    monkeypatch.setattr(loop_module, "Client", lambda server: client)
    llm = FakeLLM([
        {"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "target.py"}}}]},
        {"role": "assistant", "content": "Terminé"},
    ], generated=["ORION vaut 42 [S1]. ORION améliore les performances [S1]."])
    grounding = FakeGroundingLLM()
    verdicts = iter(["SUPPORTED", "UNSUPPORTED"])
    original = grounding.generate
    def generate(**kwargs):
        grounding.verdict = next(verdicts)
        return original(**kwargs)
    grounding.generate = generate
    agent = ToolLoop(llm=llm, grounding_llm=grounding, server=object(), memory=ConversationMemory(tmp_path))
    answer = await agent.ask("Quelle valeur ORION ?")
    assert "affirmations non supportées" in answer
    assert len(grounding.calls) == 2
    entry = json.loads((tmp_path / "history.jsonl").read_text(encoding="utf-8"))
    if debug:
        diagnostic = entry["diagnostics"]
        assert "améliore les performances" in diagnostic["candidate"]
        assert "ORION = 42" in diagnostic["sources"][0]["content"]
        assert [claim["supported"] for claim in diagnostic["claims"]] == [True, False]
        assert diagnostic["claims"][1]["source_ids"] == [1]
        assert diagnostic["claims"][1]["seconds"] >= 0
        assert "améliore les performances" not in ConversationMemory(tmp_path).recall("ORION", 4000)
    else:
        assert "diagnostics" not in entry


@pytest.mark.anyio
async def test_stdio_read_bounds_error_is_visible_and_following_call_succeeds(tmp_path, monkeypatch):
    from mcp import Client
    (tmp_path / "short.py").write_text("ORION = 42\n" * 226, encoding="utf-8")
    monkeypatch.setenv("CODEAGENT_PROJECT_PATH", str(tmp_path))
    async with Client(loop_module.build_server_parameters()) as client:
        invalid = await client.call_tool("read_file", {"path": "short.py", "start_line": 227, "end_line": 277})
        valid = await client.call_tool("read_file", {"path": "short.py", "start_line": 226, "end_line": 277})
    assert invalid.is_error is True
    assert "start_line exceeds file length (226 lines)" in loop_module.tool_result_content(invalid)
    assert valid.is_error is False
    assert json.loads(loop_module.tool_result_content(valid))["end_line"] == 226


@pytest.mark.anyio
async def test_real_mcp_taskgroup_wraps_leaf_without_losing_failed_turn(tmp_path):
    from app.mcp_server import mcp
    llm = FakeLLM([])
    def fail(**kwargs):
        raise AttributeError("reproduced MCP leaf")
    llm.chat = fail
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=mcp, memory=ConversationMemory(tmp_path))
    with pytest.raises(ExceptionGroup, match="TaskGroup"):
        await agent.ask("Question réelle MCP")
    assert "AttributeError: reproduced MCP leaf" in ConversationMemory(tmp_path).recent[0]["answer"]


def test_debug_grounding_exception_retains_attempted_claim_and_no_stale_timing(monkeypatch):
    monkeypatch.setenv("CODEAGENT_DEBUG", "1")
    grounding = FakeGroundingLLM()
    grounding.last_request_stats = {"total_duration_ms": 123}
    def fail(**kwargs):
        raise RuntimeError("Ollama timed out")
    grounding.generate = fail
    agent = ToolLoop(llm=FakeLLM([]), grounding_llm=grounding, server=object())
    with pytest.raises(RuntimeError, match="timed out"):
        agent._finalize_answer("ORION vaut 42 [S1].", [make_chunk()])
    assert agent.last_debug["claims"][0]["supported"] is None
    assert agent.last_debug["claims"][0]["ollama"] is None
    assert agent.last_debug["requests"][0]["ollama"] is None


def test_routing_audit_detects_git_diff_for_empty_conversation():
    from pathlib import Path
    from scripts.benchmark_tool_routing import run_task
    cases = json.loads((Path(__file__).resolve().parents[1] / "benchmarks/memory/routing_cases.json").read_text(encoding="utf-8"))
    llm = FakeLLM([{"role": "assistant", "tool_calls": [{"function": {"name": "get_git_diff", "arguments": {}}}]}])
    result = run_task(llm, [], cases[0])
    assert result["routing_success"] is False
    assert result["tool_selection_correct"] is False
    assert result["first_tool"] == "get_git_diff"
    assert "ORCHIDEE-9274" not in json.dumps(llm.calls[0]["messages"])


def test_routing_audit_uses_session_memory_and_keeps_mixed_tool_request():
    from pathlib import Path
    from scripts.benchmark_tool_routing import run_task
    cases = json.loads((Path(__file__).resolve().parents[1] / "benchmarks/memory/routing_cases.json").read_text(encoding="utf-8"))
    llm = FakeLLM([
        {"role": "assistant", "content": "ORCHIDEE-9274, turquoise"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "get_git_diff", "arguments": {}}}]},
    ])
    assert run_task(llm, [], cases[1])["routing_success"] is True
    assert "ORCHIDEE-9274" in json.dumps(llm.calls[0]["messages"])
    assert run_task(llm, [], cases[2])["routing_success"] is True
    assert "ORCHIDEE-9274" not in json.dumps(llm.calls[1]["messages"])
    missed_call = FakeLLM([{"role": "assistant", "content": "Ta priorité est turquoise. Je vais inspecter les modifications Git."}])
    assert run_task(missed_call, [], cases[2])["routing_success"] is False


@pytest.mark.parametrize("verdict", ["SUPPORTED", "UNSUPPORTED"])
def test_missing_citation_is_attached_only_after_single_source_verification(verdict, monkeypatch):
    monkeypatch.setenv("CODEAGENT_DEBUG", "1")
    grounding = FakeGroundingLLM(verdict)
    agent = ToolLoop(llm=FakeLLM([]), grounding_llm=grounding, server=object())
    answer = "ORION vaut 42."
    result = agent._finalize_answer(answer, [make_chunk()])
    assert len(grounding.calls) == 1
    assert agent.last_debug["candidate"] == answer
    assert agent.last_debug["citation_repair"] == ("verified" if verdict == "SUPPORTED" else "rejected")
    assert result == (answer + " [S1]" if verdict == "SUPPORTED" else
                      "La réponse générée contient des affirmations non supportées par les sources.")


@pytest.mark.parametrize("fence", ["```", "````", "~~~"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_citation_repair_preserves_code_and_inline_characters(fence, newline, monkeypatch):
    monkeypatch.setenv("CODEAGENT_DEBUG", "1")
    from app.rag.grounding import claims_have_citations

    block = (f'{fence}python\n'
             'def marker(value: str = "é\\t`[S1]`"):\n'
             '\ttext = "first\\n\\nsecond"  \n\n'
             '    return value != "\\\\" and {"key": text}\n'
             f'{fence}').replace("\n", newline)
    inline = '`value != "\\\\"`'
    candidate = f"Voici le code :{newline}{newline}{block}{newline}{newline}Expression {inline}."
    grounding = FakeGroundingLLM()
    agent = ToolLoop(llm=FakeLLM([]), grounding_llm=grounding, server=object())

    result = agent._finalize_answer(candidate, [make_chunk()])

    assert block in result
    assert inline in result
    assert result.endswith(f"Expression {inline}. [S1]")
    assert claims_have_citations(result)
    assert agent.last_grounding == "passed"
    assert agent.last_debug["citation_repair"] == "verified"


def test_citation_repair_preserves_code_blank_line_and_indentation():
    from app.agent.tool_loop import normalize_single_source_citations

    block = "```python\ndef marker():\n    value = 42\n\n    return value\n```"
    candidate = f"Voici le code :\n\n{block}\n\nLa valeur vaut 42."
    result = normalize_single_source_citations(candidate, 1)
    assert block in result
    assert result.endswith("La valeur vaut 42. [S1]")


def test_citation_repair_preserves_multiline_inline_code_and_existing_references():
    from app.agent.tool_loop import normalize_single_source_citations

    inline = '``value = "`"\n\n    value + "é"``'
    block = '```python\nvalue = "``` ~~~"\n\n    other = "é"\n```'
    candidate = f"Premier fait [S1].\n\n[S1]\n\n{block}\n\nExpression {inline}."
    assert normalize_single_source_citations(candidate, 1) == candidate.replace(
        block, block + "\n[S1]",
    ) + " [S1]"


@pytest.mark.parametrize("code", ["```python\ndef marker():\n\n    return 42", "`value\n\nmore"])
def test_unclosed_code_cannot_be_repaired_inside_the_code(code, monkeypatch):
    monkeypatch.setenv("CODEAGENT_DEBUG", "1")
    agent = ToolLoop(llm=FakeLLM([]), grounding_llm=FakeGroundingLLM(), server=object())
    result = agent._finalize_answer("Voici le code :\n\n" + code, [make_chunk()])
    assert "citations invalides ou manquantes" in result
    assert agent.last_grounding == "citation_error"
    assert agent.last_debug["citation_repair"] == "rejected"


@pytest.mark.parametrize("answer", [
    "ORION vaut 42. Je ne peux pas déterminer le reste à partir des sources disponibles.",
    "ORION vaut 42 [S9]. Je ne peux pas déterminer le reste à partir des sources disponibles.",
])
def test_partial_abstention_cannot_bypass_missing_or_invalid_references(answer):
    grounding = FakeGroundingLLM()
    agent = ToolLoop(llm=FakeLLM([]), grounding_llm=grounding, server=object())
    result = agent._finalize_answer(answer, [make_chunk(), {**make_chunk(), "file": "other.py"}])
    assert "citations invalides ou manquantes" in result
    assert agent.last_grounding == "citation_error"
    assert grounding.calls == []


@pytest.mark.anyio
async def test_failed_tool_cannot_claim_an_unexecuted_success(monkeypatch):
    monkeypatch.setattr(loop_module, "Client", lambda server: FakeClient([
        tool_result(is_error=True, content={"message": "File does not exist."}),
    ]))
    llm = FakeLLM([
        {"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "missing.py"}}}]},
        {"role": "assistant", "content": "J'ai lu et corrigé missing.py."},
    ])
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=object())
    assert await agent.ask("Lis missing.py") == "Je ne peux pas le déterminer à partir des sources disponibles."
    assert agent.last_grounding == "passed"
    assert llm.generate_calls == []


@pytest.mark.parametrize("answer", [
    "ORION vaut 42. Je ne peux pas déterminer le reste à partir des sources disponibles.",
    "ORION vaut 42 [S1].\n\nORION améliore les performances.",
])
def test_rag_partial_abstention_or_citation_does_not_skip_unsourced_claims(answer):
    llm = MagicMock()
    llm.generate.return_value = answer
    pipeline = RAGPipeline(lambda q, k: [make_chunk()], llm)
    assert "citations invalides ou manquantes" in pipeline.ask("Quelle valeur ORION ?")
    assert llm.generate.call_count == 1


def test_final_selection_records_source_mapping_and_grounds_only_presented_content(monkeypatch):
    monkeypatch.setenv("CODEAGENT_DEBUG", "1")
    from app.agent.tool_loop import build_tool_source
    sources = [
        {**make_chunk(), "content": "x" * 20000, "observation_id": "S1"},
        {**build_tool_source("read_file", {"path": "current.py"}, json.dumps({
            "path": "current.py", "start_line": 42, "end_line": 42, "content": "SCORE_8617 = 0.545",
        })), "observation_id": "S2"},
    ]
    llm = FakeLLM([], generated=["SCORE_8617 vaut 0.545 [S1]."])
    llm.context_tokens = 5500
    llm.num_predict = 300
    grounding = FakeGroundingLLM()
    agent = ToolLoop(llm=llm, grounding_llm=grounding, server=object())
    assert "0.545" in agent._generate_final_answer("Quelle valeur SCORE_8617 ?", sources, [])
    assert agent.last_debug["source_mapping"] == {"S1": "S2"}
    assert len(agent.last_debug["sources_before_selection"]) == 2
    assert "current.py" in grounding.calls[0]["prompt"]
    assert "SCORE_8617 = 0.545" in grounding.calls[0]["prompt"]
    assert "x" * 100 not in grounding.calls[0]["prompt"]


@pytest.mark.anyio
async def test_changed_project_source_is_reread_and_history_is_not_grounding_evidence(tmp_path, monkeypatch):
    from app.mcp_server import mcp
    from app.tools import project_tools
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    memory = ConversationMemory(max_tokens=1800, recent_turns=1)
    memory.record("Lis current.py", "SCORE_8617 vaut 0.544 [S1].")
    for index in range(5):
        memory.record(f"Tour secondaire {index}", "Détail. " * 40)
    (tmp_path / "current.py").write_text("SCORE_8617 = 0.545\n", encoding="utf-8")
    llm = FakeLLM([
        {"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "current.py"}}}]},
        {"role": "assistant", "content": "Terminé"},
    ], generated=["SCORE_8617 vaut 0.545 [S1]."])
    grounding = FakeGroundingLLM()
    agent = ToolLoop(llm=llm, grounding_llm=grounding, server=mcp, memory=memory)
    assert await agent.ask("Relis SCORE_8617 dans current.py") == "SCORE_8617 vaut 0.545 [S1]."
    assert '"content": "SCORE_8617 = 0.545"' in grounding.calls[0]["prompt"]
    assert "0.544" not in grounding.calls[0]["prompt"]
    assert agent.last_trace[0]["status"] == "success"


def test_multi_source_missing_citations_do_not_trigger_unproven_generation():
    llm = FakeLLM([], generated=["ORION vaut 42."])
    grounding = FakeGroundingLLM()
    agent = ToolLoop(llm=llm, grounding_llm=grounding, server=object())
    sources = [make_chunk(), {**make_chunk(), "file": "other.py"}]
    result = agent._generate_final_answer("Quelle valeur ORION ?", sources, [])
    assert len(llm.generate_calls) == 1
    assert agent.last_grounding == "citation_error"
    assert grounding.calls == []
    assert "citations invalides ou manquantes" in result


@pytest.mark.anyio
async def test_final_fitting_keeps_recovered_source_after_long_conversation(tmp_path, monkeypatch):
    from scripts.benchmark_memory import ROOT, evaluate
    from app.tools import project_tools
    monkeypatch.setattr(project_tools, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "app/target.py"
    source.parent.mkdir()
    source.write_text("ordinary = 0\n" * 599 + "ORION = 42\n", encoding="utf-8")
    case = json.loads((ROOT / "benchmarks/memory/conversations.json").read_text(encoding="utf-8"))[0]
    result = await evaluate(case, True, tmp_path / "memory", 60)
    assert result["source_valid"] is True
    assert result["tool_error_recovered"] is True
    assert result["max_context_estimate"] <= result["input_limit"]


@pytest.mark.anyio
@pytest.mark.parametrize("question", [
    "Explique-moi brièvement le rôle de FAISS dans CodeAgent.",
    "Explain the role of a component in the project.",
    "Lis app/target.py.",
    "Rappelle ma priorité dans cette session et inspecte les modifications du projet avec get_git_diff.",
])
async def test_explicit_project_question_cannot_accept_facts_without_current_tools(question, monkeypatch):
    monkeypatch.setattr(loop_module, "Client", lambda server: FakeClient())
    llm = FakeLLM([{"role": "assistant", "content": "CodeAgent fait des recommandations de code."}])
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=object())
    assert await agent.ask(question) == "Je ne peux pas le déterminer à partir des sources disponibles."
    assert agent.last_trace == []
    assert agent.last_grounding == "passed"
    assert llm.generate_calls == []


@pytest.mark.anyio
@pytest.mark.parametrize("question", [
    "Quel choix ai-je indiqué pour CodeAgent dans cette session ?",
    "Quel code ai-je choisi pour CodeAgent ?",
    "What code did I choose for CodeAgent?",
])
async def test_project_name_and_tool_in_conversation_recall_do_not_require_documentary_sources(question, monkeypatch):
    monkeypatch.setattr(loop_module, "Client", lambda server: FakeClient())
    memory = ConversationMemory()
    memory.record("Dans cette session, mon choix pour CodeAgent est read_file.", "Enregistré.")
    llm = FakeLLM([{"role": "assistant", "content": "read_file"}])
    agent = ToolLoop(llm=llm, grounding_llm=FakeGroundingLLM(), server=object(), memory=memory)
    assert await agent.ask(question) == "read_file"
    assert agent.last_grounding == "not_used"
    assert agent.last_trace == []


@pytest.mark.parametrize("candidate", [
    "```python\nx = 1\n```",
    "Avant.\n\n```python\nif True:\n    text = '<&>'\n```\n\nAprès.",
    "```python\nx = 1\n```\n\n```python\ny = 2\n```",
    "Texte `print('[S8]')`.",
    "~~~python\nx = '[S2]'\n~~~",
    "````python\ntext = '```'\n````",
    "  ```python\r\n  x = '[S8]'\r\n  ```  \r\n\r\nAprès.",
])
def test_R5_repair_preserves_markdown_structure(candidate):
    result = normalize_single_source_citations(candidate, 1)
    original = MarkdownIt().parse(candidate)
    repaired = MarkdownIt().parse(result)
    original_code = [(t.type, t.markup, t.info, t.content) for t in original if t.type == "fence"]
    repaired_code = [(t.type, t.markup, t.info, t.content) for t in repaired if t.type == "fence"]
    assert repaired_code == original_code
    assert [(c.type, c.content) for t in repaired for c in (t.children or []) if c.type == "code_inline"] == [
        (c.type, c.content) for t in original for c in (t.children or []) if c.type == "code_inline"]
    assert result.replace(" [S1]", "").replace("\r\n[S1]", "").replace("\n[S1]", "") == candidate
    assert all(not (line.strip().startswith(("```", "~~~")) and "[S1]" in line) for line in result.splitlines())
    assert all(ids == [1] for _, ids in extract_claims(result))


def test_R5_existing_reference_and_impossible_repair():
    cited = "Code [S1] :\n\n```python\n    x = 1\n```"
    assert normalize_single_source_citations(cited, 1) == cited
    for candidate in ("```python\nx = 1", "Texte `incomplet"):
        assert normalize_single_source_citations(candidate, 1) == candidate
    candidate = "```python\nx = 1\n```"
    assert normalize_single_source_citations(candidate, 2) == candidate
    judge = MagicMock()
    judge.generate.return_value = "UNSUPPORTED"
    agent = ToolLoop(llm=FakeLLM([]), grounding_llm=judge)
    assert agent._finalize_answer(candidate, [source("x = 2")]) != candidate


class Embedding:
    def encode(self, texts, **kwargs):
        return np.zeros((len(texts), 384), dtype="float32")


@pytest.mark.parametrize("state", ["coherent", "no_manifest", "no_index", "changed"])
def test_R10_index_provenance_and_restart(tmp_path, monkeypatch, state):
    import app.main as main
    from app.memory.project_state import build_manifest, save_manifest
    from app.memory.vector_store import build_index, save_index
    from app.parser.file_loader import read_dir
    from app.utils.paths import project_memory_dir
    project = tmp_path / "project"
    project.mkdir()
    file = project / "state.py"
    file.write_text("FLAG = 1\n", encoding="utf8")
    memory_root = tmp_path / "memory"
    directory = project_memory_dir(project, memory_root)
    chunks = [{"file": str(file), "type": "variable", "name": "FLAG", "content": "FLAG = 1", "start_line": 1, "end_line": 1}]
    if state != "no_index":
        save_index(build_index(Embedding(), chunks), chunks, directory)
    if state != "no_manifest":
        save_manifest(build_manifest(read_dir(project), project), directory)
    if state != "coherent":
        file.write_text("FLAG = 2\n", encoding="utf8")
    monkeypatch.setattr(main, "PROJECT_PATH", project)
    monkeypatch.setattr(main, "MEMORY_ROOT", memory_root)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda _: Embedding()))
    monkeypatch.setattr(main, "LocalLLM", lambda **kw: FakeLLM([]))
    original = main.build_index
    builder = MagicMock(wraps=original)
    monkeypatch.setattr(main, "build_index", builder)
    first, _, _ = main.build_rag_pipeline()
    second, _, _ = main.build_rag_pipeline()
    expected = "FLAG = 1" if state == "coherent" else "FLAG = 2"
    assert first.retriever.__self__.chunks[0]["content"] == expected
    assert second.retriever.__self__.chunks[0]["content"] == expected
    assert first.retriever("FLAG", 1)[0]["content"] == expected
    assert second.retriever("FLAG", 1)[0]["content"] == expected
    assert builder.call_count == (1 if state in {"no_index", "no_manifest"} else 0)


@pytest.mark.parametrize("failure", ["build", "serialize", "interrupt_index", "interrupt_chunks", "interrupt_manifest"])
def test_R10_failed_update_never_marks_old_index_current(tmp_path, monkeypatch, failure):
    import app.main as main
    from pathlib import Path
    from app.memory.project_state import load_manifest
    from app.memory.vector_store import load_index
    project = tmp_path / "project"
    project.mkdir()
    file = project / "state.py"
    file.write_text("FLAG = 1\n", encoding="utf8")
    monkeypatch.setattr(main, "PROJECT_PATH", project)
    monkeypatch.setattr(main, "MEMORY_ROOT", tmp_path / "memory")
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda _: Embedding()))
    monkeypatch.setattr(main, "LocalLLM", lambda **kw: FakeLLM([]))
    _, _, directory = main.build_rag_pipeline()
    old = {name: (directory / name).read_bytes() for name in ("index.faiss", "chunks.json", "manifest.json")}
    file.write_text("FLAG = 2\n", encoding="utf8")
    with monkeypatch.context() as fault:
        if failure == "build":
            fault.setattr(main, "update_index", MagicMock(side_effect=RuntimeError("embedding failed")))
        elif failure == "serialize":
            fault.setattr(main, "save_manifest", MagicMock(side_effect=OSError("disk full")))
        else:
            replace = Path.replace
            filename = {"interrupt_index": "index.faiss", "interrupt_chunks": "chunks.json", "interrupt_manifest": "manifest.json"}[failure]
            def interrupted(path, target):
                if path.name == filename:
                    raise KeyboardInterrupt("interrupted publication")
                return replace(path, target)
            fault.setattr(Path, "replace", interrupted)
        with pytest.raises((RuntimeError, OSError, KeyboardInterrupt)):
            main.build_rag_pipeline()
    if failure in {"build", "serialize"}:
        assert all((directory / name).read_bytes() == content for name, content in old.items())
        assert load_index(directory)[1][0]["content"] == "FLAG = 1"
    else:
        assert not (directory / "manifest.json").exists()
    repaired, _, _ = main.build_rag_pipeline()
    assert repaired.retriever.__self__.chunks[0]["content"] == "FLAG = 2"
    assert repaired.retriever("FLAG", 1)[0]["content"] == "FLAG = 2"
    assert "state.py" in load_manifest(directory)
    restarted, _, _ = main.build_rag_pipeline()
    assert restarted.retriever.__self__.chunks[0]["content"] == "FLAG = 2"
    assert restarted.retriever("FLAG", 1)[0]["content"] == "FLAG = 2"
