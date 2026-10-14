import json

import pytest

from app.memory.context import ContextBudget, estimate_tokens, reduce_observation
from app.memory.conversation import ConversationMemory
from app.utils.paths import project_memory_dir


def test_recent_roles_and_followup_reference():
    memory = ConversationMemory()
    budget = ContextBudget()
    empty = memory.prepare("Bonjour", "System", budget)
    assert [message["role"] for message in empty] == ["system", "user"]
    memory.record("Décision : nom ORION", "Nom enregistré")
    memory.record("Ne pas ajouter de dépendance", "Contrainte enregistrée")
    messages = memory.prepare("Quel nom avons-nous choisi ?", "System", budget)
    recent = messages[-5:]
    assert [message["role"] for message in recent] == ["user", "assistant", "user", "assistant", "user"]
    assert recent[0]["content"] == "Décision : nom ORION"
    assert recent[-1]["content"] == "Quel nom avons-nous choisi ?"
    assert memory.compressions == 0


def test_sessions_projects_and_legacy_history_are_independent(tmp_path):
    first_dir = project_memory_dir(tmp_path / "a", tmp_path / "memory")
    second_dir = project_memory_dir(tmp_path / "b", tmp_path / "memory")
    first = ConversationMemory(first_dir, "one")
    first.record("Décision ORION", "OK")
    with (first_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"question": "Ancien secret", "answer": "M4", "timestamp": "old"}) + "\n")
    assert "ORION" in ConversationMemory(first_dir, "one").recall("Décision")
    assert "secret" not in ConversationMemory(first_dir, "one").recall("secret")
    assert ConversationMemory(first_dir, "two").recall("ORION") == ""
    assert ConversationMemory(second_dir, "one").recall("ORION") == ""


@pytest.mark.parametrize("session", ["../escape", "", "a/b", "a" * 65])
def test_invalid_session_id_rejected(session):
    with pytest.raises(ValueError, match="Session ID"):
        ConversationMemory(session_id=session)


def test_compaction_threshold_and_priority_fidelity(tmp_path):
    memory = ConversationMemory(tmp_path, "fidelity", max_tokens=1800, recent_turns=1)
    memory.record("Décision : cible ORION. Ne pas ajouter de dépendance. Référence app/core.py:42.", "Hypothèse non validée : cache indisponible.")
    assert memory.compressions == 0
    for number in range(8):
        memory.record(f"Tour {number}", "Détails sans importance. " * 30)
    assert memory.compressions > 0
    assert memory.compacted > 0
    assert len(memory.recent) == 1
    restored = ConversationMemory(tmp_path, "fidelity", max_tokens=1800, recent_turns=1)
    recall = restored.recall("ORION dépendance référence cache hypothèse", 3000)
    for fact in ["ORION", "Ne pas ajouter de dépendance", "app/core.py:42", "Hypothèse non validée"]:
        assert fact in recall
    assert restored.compacted == memory.compacted
    assert restored.compressions == memory.compressions
    state = json.loads(memory.state_path.read_text(encoding="utf-8"))
    keys = [(item["role"], item["content"]) for item in state["summary"]]
    assert len(keys) == len(set(keys))
    assert estimate_tokens(state["summary"]) <= memory.max_tokens // 2


def test_updated_and_contradictory_information_keeps_chronology():
    memory = ConversationMemory(max_tokens=1800, recent_turns=1)
    memory.record("Décision : cible ORION", "Hypothèse : ORION convient")
    memory.record("Décision mise à jour : cible VEGA, ORION abandonné", "OK")
    for number in range(4):
        memory.record(f"Tour {number}", "Texte neutre. " * 25)
    recall = memory.recall("cible ORION VEGA", 3000)
    assert "Décision : cible ORION" in recall
    assert "Décision mise à jour : cible VEGA, ORION abandonné" in recall
    items = json.loads(recall.split("\n", 1)[1])
    assert [item["turn"] for item in items] == sorted(item["turn"] for item in items)


def test_reaffirmed_decision_keeps_latest_turn_after_compaction_and_restart(tmp_path):
    memory = ConversationMemory(tmp_path, "reaffirmed", max_tokens=1800, recent_turns=1)
    for target in ("ORION", "VEGA", "ORION"):
        memory.record(f"Décision cible {target}.", "OK")
    for number in range(8):
        memory.record(f"Tour {number}", "Texte neutre. " * 40)
    assert memory.compacted >= 3
    restored = ConversationMemory(tmp_path, "reaffirmed", max_tokens=1800, recent_turns=1)
    items = json.loads(restored.recall("Décision cible", 3000).split("\n", 1)[1])
    decisions = [item for item in items if item["role"] == "user" and "Décision cible" in item["content"]]
    assert decisions[-1]["content"] == "Décision cible ORION."
    assert decisions[-1]["turn"] == 3


def test_legacy_checkpoint_rebuilds_reaffirmed_decision_from_archive(tmp_path):
    memory = ConversationMemory(tmp_path, "legacy", max_tokens=1800, recent_turns=1)
    for target in ("ORION", "VEGA", "ORION"):
        memory.record(f"Décision cible {target}.", "OK")
    # Valid M7 v1 cache produced by the old cross-turn deduplication.
    memory.state_path.write_text(json.dumps({"version": 1, "compacted": 3, "compressions": 1, "summary": [
        {"turn": 1, "role": "user", "content": "Décision cible ORION.", "priority": True},
        {"turn": 2, "role": "user", "content": "Décision cible VEGA.", "priority": True},
    ]}), encoding="utf-8")
    restored = ConversationMemory(tmp_path, "legacy", max_tokens=1800, recent_turns=1)
    items = json.loads(restored.recall("Quel dernier choix ?", 2000).split("\n", 1)[1])
    decisions = [item for item in items if item["role"] == "user" and "Décision cible" in item["content"]]
    assert decisions[-1]["content"] == "Décision cible ORION."
    assert decisions[-1]["turn"] == 3


def test_compaction_preserves_exact_values_paths_lines_and_corrections(tmp_path):
    facts = [
        "Décision : SCORE_8617 vaut 0.544, chemin app/core.py:227-277.",
        "Correction : SCORE_8617 vaut 0.545, chemin app/current.py:200-226. Ne pas ajouter de dépendance.",
    ]
    memory = ConversationMemory(tmp_path, "exact", max_tokens=2400, recent_turns=1)
    for fact in facts:
        memory.record(fact, "Hypothèse non validée.")
    for number in range(8):
        memory.record(f"Tour {number}", "Texte neutre. " * 40)
    restored = ConversationMemory(tmp_path, "exact", max_tokens=2400, recent_turns=1)
    items = json.loads(restored.recall("SCORE_8617 chemin dépendance", 3000).split("\n", 1)[1])
    for turn, fact in enumerate(facts, 1):
        excerpts = [item["content"] for item in items if item["turn"] == turn and item["role"] == "user"]
        assert " ".join(excerpts) == fact


def test_multiple_active_observations_share_budget_and_keep_source_ids():
    from app.agent.tool_loop import TOOL_SYSTEM_PROMPT
    budget = ContextBudget(7000, 300)
    active = []
    for index in range(5):
        arguments = {"path": f"app/target_{index}.py"}
        active.append({"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "read_file", "arguments": arguments}},
        ]})
        raw = {"path": arguments["path"], "start_line": 10, "end_line": 609,
               "content": "ordinary = 0\n" * 599 + "ORION = 0.545"}
        active.append({"role": "tool", "tool_name": "read_file", "content": json.dumps({
            "ok": True, "tool": "read_file", "source": f"[S{index + 1}]", "result": json.dumps(raw),
        })})
    original = json.dumps(active)
    messages = ConversationMemory().prepare("Lis ORION", TOOL_SYSTEM_PROMPT, budget, active=active)
    assert budget.measure(messages) <= budget.input_limit
    selected = [message for message in messages if message["role"] == "tool"]
    assert len(selected) == 5
    for index, message in enumerate(selected, 1):
        assert json.loads(message["content"])["source"] == f"[S{index}]"
        assert "ORION = 0.545" in message["content"]
        assert '"line": 609' in message["content"]
    assert json.dumps(active) == original


def test_extraction_failure_preserves_archive_and_recent_turn(tmp_path, monkeypatch):
    memory = ConversationMemory(tmp_path, max_tokens=800, recent_turns=1)
    memory.record("Décision ORION", "x" * 500)
    def fail(*args):
        raise RuntimeError("compression unavailable")
    monkeypatch.setattr(memory, "_extract", fail)
    with pytest.raises(RuntimeError, match="compression unavailable"):
        memory.record("Deuxième tour", "y" * 500)
    assert memory.recent[0]["question"] == "Décision ORION"
    assert len(list(memory._entries())) == 2
    restored = ConversationMemory(tmp_path, max_tokens=800, recent_turns=1)
    assert restored.total_turns == 2
    assert "ORION" in restored.recall("ORION", 1800)


def test_corrupt_checkpoint_and_partial_jsonl_recover_without_overwrite(tmp_path):
    memory = ConversationMemory(tmp_path, "restore")
    memory.record("Contrainte : conserver ORION", "OK")
    memory.state_path.write_text('{"broken":', encoding="utf-8")
    with (tmp_path / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"session_id": "restore",')
    with pytest.warns(RuntimeWarning, match="rebuilding"):
        restored = ConversationMemory(tmp_path, "restore")
    restored.record("Nouvelle décision VEGA", "OK")
    again = ConversationMemory(tmp_path, "restore")
    assert again.total_turns == 2
    assert "ORION" in again.recall("ORION")
    assert "VEGA" in again.recall("VEGA")
    assert '{"broken":' not in again.state_path.read_text(encoding="utf-8")
    assert '{"session_id": "restore",' in (tmp_path / "history.jsonl").read_text(encoding="utf-8")


def test_checkpoint_ahead_of_archive_fails_closed(tmp_path):
    memory = ConversationMemory(tmp_path)
    memory.record("ORION", "OK")
    state = {"version": 1, "compacted": 5, "compressions": 1, "summary": []}
    memory.state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="shorter than its checkpoint"):
        ConversationMemory(tmp_path)
    assert json.loads(memory.state_path.read_text()) == state


def test_multiple_write_restore_cycles_and_old_selective_recall(tmp_path):
    for number in range(15):
        memory = ConversationMemory(tmp_path, max_tokens=1200, recent_turns=1)
        memory.record(f"Décision identifiant ITEM_{number} : valeur {number}", "Détails. " * 40)
    restored = ConversationMemory(tmp_path, max_tokens=1200, recent_turns=1)
    assert restored.total_turns == 15
    recall = restored.recall("ITEM_0", 1600)
    assert "ITEM_0" in recall
    assert estimate_tokens(recall) <= 1600
    assert "ITEM_1 :" not in recall


def test_observations_preserve_role_order_and_historical_provenance():
    observations = [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "app/core.py"}}}]},
        {"role": "tool", "tool_name": "read_file", "content": '{"path":"app/core.py","start_line":42,"result":"ORION","source":"[S1]"}'},
    ]
    memory = ConversationMemory()
    memory.record("Lire app/core.py", "ORION [S1]", observations)
    messages = memory.prepare("Quelle valeur ORION dans app/core.py ?", "System", ContextBudget())
    recent = messages[-5:]
    assert [message["role"] for message in recent] == ["user", "assistant", "tool", "assistant", "user"]
    assert "app/core.py" in recent[2]["content"]
    assert "42" in recent[2]["content"]
    assert "[S1]" not in json.dumps(messages)
    assert observations[1]["content"].endswith('"[S1]"}')


def test_budget_exact_boundary_overflow_schema_and_utf8():
    messages = [{"role": "system", "content": "é漢"}, {"role": "user", "content": "Q"}]
    tools = [{"name": "read_file", "description": "x" * 100}]
    size = ContextBudget().measure(messages, tools)
    exact = ContextBudget(size + 512 + 256, 512)
    assert exact.check(messages, tools) == exact.input_limit
    assert estimate_tokens("é漢") == 5
    with pytest.raises(RuntimeError, match="Context budget exceeded"):
        ContextBudget(size + 512 + 255).check(messages, tools)
    with pytest.raises(ValueError):
        ContextBudget(256)


def test_long_conversation_context_remains_bounded_and_current_question_intact():
    memory = ConversationMemory(max_tokens=1200, recent_turns=2)
    budget = ContextBudget(2800, 300)
    for number in range(100):
        memory.record(f"Tour {number} ; décision ORION", "Explication. " * 25)
        messages = memory.prepare("Rappelle ORION", "Instructions prioritaires", budget)
        assert budget.measure(messages) <= budget.input_limit
        assert messages[0]["content"] == "Instructions prioritaires"
        assert messages[-1]["content"] == "Rappelle ORION"
    with pytest.raises(RuntimeError, match="Context budget exceeded"):
        memory.prepare("x" * 20_000, "System", budget)


def test_structured_large_observation_selects_relevant_item_without_invention():
    matches = [{"file": f"app/file_{number}.py", "line": number + 1, "text": "ordinary"} for number in range(100)]
    matches[57] = {"file": "app/target.py", "line": 58, "text": "ORION = 42"}
    content = json.dumps({"matches": matches, "truncated": False})
    reduced = reduce_observation(content, "ORION target", 650)
    data = json.loads(reduced)
    assert data["matches"][0] == matches[57]
    assert all(item in matches for item in data["matches"])
    assert data["truncated"] is True
    assert estimate_tokens(reduced) <= 650
    assert reduce_observation('{"value":42}', "value") == '{"value":42}'


def test_file_excerpts_preserve_exact_line_numbers_and_values():
    lines = [f"ordinary_{number} = {number}" for number in range(100)]
    lines[50] = "ORION = 42"
    content = json.dumps({"path": "app/target.py", "start_line": 10, "end_line": 109, "content": "\n".join(lines)})
    reduced = json.loads(reduce_observation(content, "ORION", 650))
    assert reduced["path"] == "app/target.py"
    assert {"line": 60, "text": "ORION = 42"} in reduced["content_excerpts"]
    assert all(item["text"] == lines[item["line"] - 10] for item in reduced["content_excerpts"])


def test_nested_tool_observation_remains_valid_json_and_bounded():
    content = json.dumps({"ok": True, "tool": "read_file", "source": "[S1]", "result": json.dumps({
        "path": "app/target.py", "start_line": 10, "content": "ordinary\n" * 200 + "ORION = 42"
    })})
    reduced = reduce_observation(content, "ORION", 750)
    assert estimate_tokens(reduced) <= 750
    assert "ORION = 42" in reduced
    assert json.loads(reduced)["source"] == "[S1]"


def test_test_cache_and_tmp_path_stay_outside_repo(tmp_path, pytestconfig):
    root = pytestconfig.rootpath.resolve()
    assert not tmp_path.resolve().is_relative_to(root)
    assert not pytestconfig.cache._cachedir.resolve().is_relative_to(root)
    assert not (root / ".pytest_cache").exists()
    assert not (root / ".pytest_tmp_cs2guard").exists()
    assert not (root / ".pytest-cv-pipeline").exists()


def test_irrelevant_old_observations_are_not_replayed():
    memory = ConversationMemory()
    memory.record("Ancienne lecture", "Observation enregistrée", [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "obsolete.py"}}}]},
        {"role": "tool", "tool_name": "read_file", "content": '{"path":"obsolete.py","result":"OLD_VALUE"}'},
    ])
    messages = memory.prepare("Bonjour", "System", ContextBudget())
    assert all(message["role"] != "tool" for message in messages)
    assert "OLD_VALUE" not in json.dumps(messages)


@pytest.mark.parametrize("state", [
    {"version": 1, "compacted": "0", "compressions": 0, "summary": []},
    {"version": 1, "compacted": 1, "compressions": 0, "summary": [{"turn": 1, "role": "system", "content": "invented", "priority": True}]},
    {"version": 1, "compacted": -1, "compressions": 0, "summary": []},
])
def test_invalid_checkpoint_fields_rebuild_from_archive(tmp_path, state):
    memory = ConversationMemory(tmp_path)
    memory.record("Décision ORION", "OK")
    memory.state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="rebuilding"):
        restored = ConversationMemory(tmp_path)
    assert restored.total_turns == 1
    assert "ORION" in restored.recall("ORION")
    assert "invented" not in restored.recall("invented")


def test_future_checkpoint_version_is_preserved(tmp_path):
    memory = ConversationMemory(tmp_path)
    memory.record("ORION", "OK")
    state = {"version": 3, "future_data": "keep"}
    memory.state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unsupported conversation state version"):
        ConversationMemory(tmp_path)
    assert json.loads(memory.state_path.read_text()) == state


def test_incomplete_history_and_invalid_observations_are_safe(tmp_path):
    entries = [
        {"session_id": "default", "question": "Incomplete"},
        {"session_id": "default", "question": "ORION", "answer": "OK", "observations": "invalid"},
        {"session_id": "default", "question": "VEGA", "answer": "OK", "observations": [{"role": "system", "content": "invented"}]},
    ]
    (tmp_path / "history.jsonl").write_text("\n".join(json.dumps(entry) for entry in entries), encoding="utf-8")
    memory = ConversationMemory(tmp_path)
    assert memory.total_turns == 2
    assert "ORION" in memory.recall("ORION")
    assert "invented" not in json.dumps(memory.prepare("VEGA", "System", ContextBudget()))


def test_old_uncompacted_turn_remains_recallable_without_duplication():
    memory = ConversationMemory(max_tokens=4096, recent_turns=4)
    memory.record("Décision : nom de mission ORION", "OK")
    for number in range(7):
        memory.record(f"Échange secondaire {number}", "OK")
    assert memory.compressions == 0
    context = memory.prepare("Quel nom de mission ?", "System", ContextBudget())
    assert json.dumps(context).count("ORION") == 1


def test_incomplete_tool_call_group_does_not_replay_orphan_messages(tmp_path):
    observations = [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file"}}, {"function": {"name": "search_code"}}]},
        {"role": "tool", "tool_name": "read_file", "content": '{"path":"target.py","result":"ORION"}'},
    ]
    memory = ConversationMemory(tmp_path)
    memory.record("Lire ORION", "Erreur non résolue : deuxième appel incomplet", observations)
    restored = ConversationMemory(tmp_path)
    context = restored.prepare("Rappelle l'erreur ORION", "System", ContextBudget())
    assert not any(message.get("tool_calls") or message["role"] == "tool" for message in context)
    assert "Erreur non résolue" in json.dumps(context, ensure_ascii=False)
    assert len(json.loads((tmp_path / "history.jsonl").read_text(encoding="utf-8"))["observations"]) == 2


def test_repeated_compaction_restart_isolates_sessions_and_same_named_projects(tmp_path):
    directories = [project_memory_dir(tmp_path / project / "same", tmp_path / "memory") for project in ("a", "b")]
    facts = {
        (0, "one"): "Décision ORION. Ne pas ajouter de dépendance. Référence app/core.py:42. Hypothèse non validée : CACHE_COLD.",
        (0, "two"): "Décision VEGA. Contrainte bibliothèque native. Référence app/other.py:17.",
        (1, "one"): "Décision SIRIUS. Contrainte hors réseau. Référence lib/source.py:9.",
        (1, "two"): "Décision ALTAIR. Contrainte stockage local. Référence lib/other.py:8.",
    }
    for (project, session), fact in facts.items():
        ConversationMemory(directories[project], session, max_tokens=1200, recent_turns=1).record(fact, "Enregistré.")
    for cycle in range(12):
        for project, session in facts:
            memory = ConversationMemory(directories[project], session, max_tokens=1200, recent_turns=1)
            memory.record(f"Tour secondaire {cycle}", "Détail temporaire. " * 40)
    for key, fact in facts.items():
        project, session = key
        memory = ConversationMemory(directories[project], session, max_tokens=1200, recent_turns=1)
        assert memory.compressions >= 10
        assert memory.compacted >= 10
        recall = memory.recall(fact, 3000)
        for sentence in fact.split(". "):
            assert sentence in recall
        context = memory.prepare(fact, "System", ContextBudget(5000))
        text = json.dumps(context, ensure_ascii=False)
        for other, other_fact in facts.items():
            if other != key:
                assert other_fact.split()[1].rstrip(".") not in text
                assert other_fact.split()[1].rstrip(".") not in recall
        assert memory.total_turns == 13
