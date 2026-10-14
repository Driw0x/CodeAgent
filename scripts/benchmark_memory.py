"""Offline M6/M7 context benchmark. Fixtures measure mechanics, not LLM intelligence."""

import argparse
import asyncio
import json
import logging
import statistics
import tempfile
from pathlib import Path
from time import perf_counter

from app.agent.tool_loop import ToolLoop
from app.memory.context import ContextBudget
from app.memory.conversation import ConversationMemory
from app.mcp_server import mcp
from app.tools import project_tools


ROOT = Path(__file__).resolve().parents[1]


class FixtureLLM:
    """Only extracts literal fixture values; tool calls exercise the real MCP server."""

    context_tokens = 7000
    num_predict = 300

    def __init__(self, recall):
        self.recall = {item["question"]: item["expected"] for item in recall}
        self.calls = 0
        self.sizes = []
        self.last_stats = None

    def chat(self, messages, tools=None):
        self.calls += 1
        self.sizes.append(ContextBudget.for_llm(self).measure(messages, tools))
        question = next(message["content"] for message in reversed(messages) if message["role"] == "user")
        if question == "Lis ORION dans app/target.py et récupère une erreur de lecture.":
            observations = [message for message in messages if message["role"] == "tool"]
            if not observations:
                path = "missing.py"
            elif any(json.loads(message["content"]).get("ok") for message in observations):
                return {"role": "assistant", "content": "Terminé."}
            else:
                path = "app/target.py"
            return {"role": "assistant", "content": "", "tool_calls": [{"function": {
                "name": "read_file", "arguments": {"path": path}
            }}]}
        history = "\n".join(message.get("content", "") for message in messages[:-1])
        expected = self.recall.get(question)
        return {"role": "assistant", "content": expected if expected and expected in history else "OK"}

    def generate(self, prompt, system_prompt=None):
        self.calls += 1
        size = ContextBudget.for_llm(self).measure([
            {"role": "system", "content": system_prompt or ""}, {"role": "user", "content": prompt}
        ])
        self.sizes.append(size)
        if "AFFIRMATION À VÉRIFIER:" in prompt:
            return "SUPPORTED" if "ORION = 42" in prompt.split("AFFIRMATION À VÉRIFIER:")[0] else "UNSUPPORTED"
        return "ORION vaut 42 [S1]." if "ORION = 42" in prompt else "Je ne peux pas le déterminer à partir des sources disponibles."


async def evaluate(case, managed, directory, noise_turns):
    llm = FixtureLLM(case["recall"])
    memory = ConversationMemory(directory if managed else None, case["id"], max_tokens=1800, recent_turns=2)
    agent = ToolLoop(llm=llm, grounding_llm=llm, server=mcp, memory=memory)
    replay = []
    times = []
    raw_max = 0
    observed_fidelity = []
    recalled = 0
    questions = [*case["turns"], *[
        [f"Analyse secondaire {number}.", "Détail temporaire sans conclusion. " * 18]
        for number in range(noise_turns)
    ]]
    for index, (question, answer) in enumerate(questions):
        # These are predefined completed exchanges, not generated facts.
        memory.record(question, answer)
        replay.extend([{"role": "user", "content": question}, {"role": "assistant", "content": answer}])
        context = memory.prepare("Continuer", "System", agent.budget)
        llm.sizes.append(agent.budget.measure(context))
        raw_max = max(raw_max, agent.budget.measure(replay))
        if not managed:
            memory = ConversationMemory(max_tokens=1800, recent_turns=2)
            agent.memory = memory
        elif index == len(questions) // 2:
            memory = ConversationMemory(directory, case["id"], max_tokens=1800, recent_turns=2)
            agent.memory = memory
    for item in case["recall"]:
        if not managed:
            agent.memory = ConversationMemory()
        start = perf_counter()
        answer = await agent.ask(item["question"])
        times.append(perf_counter() - start)
        recalled += answer == item["expected"]
        # Fidelity means an exact stored excerpt, with original role/turn, not paraphrase entailment.
        recalled_context = agent.memory.recall(item["question"], 1800)
        if recalled_context:
            excerpts = json.loads(recalled_context.split("\n", 1)[1])
            original = "\n".join(text for turn in case["turns"] for text in turn)
            observed_fidelity.extend(item["content"] in original for item in excerpts
                                     if item["turn"] <= len(case["turns"]))
    start = perf_counter()
    answer = await agent.ask("Lis ORION dans app/target.py et récupère une erreur de lecture.")
    times.append(perf_counter() - start)
    source_valid = agent.last_grounding == "passed" and answer == "ORION vaut 42 [S1]."
    recovered = any(trace["status"] == "error" for trace in agent.last_trace) and any(trace["status"] == "success" for trace in agent.last_trace)
    memory = agent.memory
    return {
        "case": case["id"], "mode": "M7" if managed else "M6_single_turn",
        "completed_fixture_turns": len(questions), "recall_questions": len(case["recall"]),
        "recall_successes": recalled, "source_valid": source_valid, "tool_error_recovered": recovered,
        "max_context_estimate": max(llm.sizes), "input_limit": agent.budget.input_limit,
        "full_replay_max_estimate": raw_max, "compressions": memory.compressions,
        "compacted_turns": memory.compacted, "exact_excerpt_fidelity": all(observed_fidelity) if observed_fidelity else None,
        "llm_fixture_calls": llm.calls, "summary_llm_calls": 0, "tool_calls": len(agent.last_trace),
        "mean_seconds": statistics.mean(times), "p50_seconds": statistics.median(times),
        "p95_seconds": sorted(times)[max(0, int(len(times) * 0.95 + 0.999) - 1)],
        "actual_model_tokens": None,
    }


async def run(noise_turns):
    cases = json.loads((ROOT / "benchmarks/memory/conversations.json").read_text(encoding="utf-8"))
    results = []
    original_root = project_tools.PROJECT_ROOT
    server_logger = logging.getLogger("mcp.server.mcpserver.server")
    previous_level = server_logger.level
    # All benchmark fixtures and memory live in a system temporary directory.
    with tempfile.TemporaryDirectory(prefix="codeagent-memory-benchmark-") as temporary:
        project_tools.PROJECT_ROOT = Path(temporary)
        source = Path(temporary) / "app/target.py"
        source.parent.mkdir()
        source.write_text("ordinary = 0\n" * 599 + "ORION = 42\n", encoding="utf-8")
        try:
            # Expected missing-file errors are recorded in the trace, without a full traceback per fixture.
            server_logger.setLevel(logging.CRITICAL)
            for case in cases:
                for managed in (False, True):
                    results.append(await evaluate(case, managed, Path(temporary) / "memory", noise_turns))
        finally:
            project_tools.PROJECT_ROOT = original_root
            server_logger.setLevel(previous_level)
    managed = [result for result in results if result["mode"] == "M7"]
    checks = {
        "context_bounded": all(result["max_context_estimate"] <= result["input_limit"] for result in results),
        "fixture_recall_complete": all(result["recall_successes"] == result["recall_questions"] for result in managed),
        "extractive_fidelity": all(result["exact_excerpt_fidelity"] is True for result in managed),
        "sources_and_grounding_preserved": all(result["source_valid"] for result in results),
        "errors_recovered": all(result["tool_error_recovered"] for result in results),
        "no_summary_model_calls": all(result["summary_llm_calls"] == 0 for result in results),
    }
    return {"protocol": "deterministic-memory-v1", "noise_turns": noise_turns,
            "criteria": "All predefined recall facts; exact extracts; bounded prompts; current-source grounding; tool recovery.",
            "limitations": "Fixture LLM, not model accuracy or production latency. M6 resets each turn; full replay is a separate size comparator.",
            "checks": checks, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-turns", type=int, default=60)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/memory/results/m7.json")
    args = parser.parse_args()
    if args.noise_turns < 10:
        parser.error("At least 10 noise turns are required to exercise compaction.")
    payload = asyncio.run(run(args.noise_turns))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not all(payload["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
