"""Fixed Ollama validation; keep autonomous runs and fixed-source replays separate."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

from mcp import Client

from app.agent.tool_loop import ToolLoop, build_server_parameters, build_tool_source, tool_result_content
from app.memory.conversation import ConversationMemory
from app.rag.citations import extract_citations, is_abstention

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks/memory/final_cases.json"


def snapshot():
    try:
        with urlopen("http://127.0.0.1:11434/api/ps", timeout=10) as response:
            return json.load(response)
    except Exception as error:
        return {"error": str(error)}


async def measure(agent, case, mode, repetition=1, sources=None, observations=None):
    before = snapshot()
    start = perf_counter()
    error = None
    answer = ""
    try:
        if case["id"] == "S8":
            answer = agent._finalize_answer(case["candidate"], sources)
        elif sources is not None:
            answer = agent._generate_final_answer(case["question"], sources, observations or [])
        else:
            answer = await agent.ask(case["question"])
    except Exception as failure:
        from app.memory.history import describe_error
        error = describe_error(failure)
    seconds = perf_counter() - start
    agent._refresh_token_stats()
    trace = agent.last_trace
    success = error is None and all(value.casefold() in answer.casefold() for value in case.get("expected", []))
    if case["id"] in {"S1", "S2", "S3", "S4", "S7"}:
        success &= not trace and agent.last_grounding == "not_used"
    if case["id"] == "S2":
        success &= agent.memory.compressions > 0
    if case["id"] == "S4":
        success &= not any(value in answer for value in case["forbidden"])
        success &= any(value in answer.casefold() for value in ("pas", "aucun", "inconnu", "absent"))
    if case["id"] in {"S5", "S6"}:
        success &= agent.last_grounding == "passed" and bool(extract_citations(answer)) and not is_abstention(answer)
        if mode == "autonomous":
            success &= any(item["status"] == "success" for item in trace)
    if case["id"] == "S6":
        success &= any(item["status"] == "error" for item in trace)
        success &= any(item["tool"] == "read_file" and item["status"] == "success" for item in trace)
    if case["id"] == "S8":
        success &= agent.last_grounding == "unsupported"
    return {
        "id": case["id"], "mode": mode, "repetition": repetition, "success": bool(success),
        "question": case.get("question"), "candidate_input": case.get("candidate"),
        "answer": answer, "error": error, "grounding": agent.last_grounding,
        "seconds": seconds, "trace": trace, "stage_seconds": agent.last_stage_times,
        "tokens": agent.last_token_stats, "diagnostics": agent.last_debug,
        "memory": getattr(agent.memory, "last_stats", {}),
        "compressions": agent.memory.compressions, "ollama_before": before, "ollama_after": snapshot(),
    }


async def fixed_sources(protocol):
    """Every replay source is obtained through actual MCP, once, then kept unchanged."""
    sources = []
    read_case = None
    async with Client(build_server_parameters()) as client:
        for arguments in next(case["reads"] for case in protocol["scenarios"] if case["id"] == "S5"):
            result = await client.call_tool("read_file", arguments)
            if result.is_error:
                raise RuntimeError(tool_result_content(result))
            sources.append(build_tool_source("read_file", arguments, tool_result_content(result)))
            sources[-1]["observation_id"] = f"S{len(sources)}"
        path = "app/retrieval/faiss_retriever.py"
        arguments = {"path": path, "start_line": 227, "end_line": 277}
        failure = await client.call_tool("read_file", arguments)
        if not failure.is_error:
            raise RuntimeError("The fixed S6 range is no longer invalid; revise the protocol before running.")
        length = len((ROOT / path).read_text(encoding="utf-8").splitlines())
        corrected = {"path": path, "start_line": max(1, length - 26), "end_line": length}
        result = await client.call_tool("read_file", corrected)
        if result.is_error:
            raise RuntimeError(tool_result_content(result))
        raw = tool_result_content(result)
        error = json.dumps({"ok": False, "error": "tool_error", "tool": "read_file",
                            "message": tool_result_content(failure)}, ensure_ascii=False)
        read_case = ([build_tool_source("read_file", corrected, raw)], [
            {"role": "tool", "tool_name": "read_file", "content": error},
            {"role": "tool", "tool_name": "read_file", "content": raw},
        ], [{"tool": "read_file", "arguments": arguments, "status": "error"},
            {"tool": "read_file", "arguments": corrected, "status": "success"}])
    return sources, read_case


async def run(args, protocol):
    os.environ["CODEAGENT_DEBUG"] = "1"
    os.environ["CODEAGENT_PROJECT_PATH"] = str(ROOT)
    cases = {case["id"]: case for case in protocol["scenarios"]}
    if args.restart_memory:
        agent = ToolLoop(memory=ConversationMemory(args.restart_memory, "memory"))
        return await measure(agent, cases["S3"], "process_restart")
    payload = {"protocol": protocol, "results": [], "started_ollama": snapshot(),
               "models": ["devstral:24b", "qwen2.5-coder:14b"],
               "limitations": "Success flags need factual review of candidates and claims. No latency aggregation across rejection/acceptance paths."}

    def save(result):
        payload["results"].append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("id", "mode", "repetition", "success", "seconds", "error")}, ensure_ascii=False), flush=True)

    with tempfile.TemporaryDirectory(prefix="codeagent-m7-final-") as temporary:
        directory = Path(temporary)
        if args.mode in {"all", "memory"}:
            memory = ConversationMemory(directory, "memory")
            agent = ToolLoop(memory=memory)
            seed = {"id": "seed", "question": protocol["memory_seed"]}
            save(await measure(agent, seed, "memory_setup"))
            for question in protocol["intermediate_questions"]:
                save(await measure(agent, {"id": "intermediate", "question": question}, "memory_setup"))
            save(await measure(agent, cases["S1"], "autonomous"))
            child = subprocess.run([sys.executable, "-m", "scripts.benchmark_m7_final", "--restart-memory", str(directory)],
                                   cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            if child.returncode:
                raise RuntimeError(child.stderr)
            save(json.loads(child.stdout))
            save(await measure(ToolLoop(memory=ConversationMemory(directory, "isolated")), cases["S4"], "autonomous"))
            compact = ConversationMemory(directory, "compressed", max_tokens=2400, recent_turns=1)
            for question, answer in protocol["compression_fixture"]:
                compact.record(question, answer)
            for number in range(12):
                compact.record(f"Échange secondaire {number}.", "Détail temporaire sans conclusion. " * 25)
            save(await measure(ToolLoop(memory=compact), cases["S2"], "fixture_history_real_answer"))
            routing = ConversationMemory()
            routing.record("Mon code de test est ORCHIDEE-9274 et ma priorité est turquoise.", "Informations retenues.")
            save(await measure(ToolLoop(memory=routing), cases["S7"], "autonomous"))
        if args.mode in {"all", "finals", "unsupported"}:
            sources, read_case = await fixed_sources(protocol)
            payload["fixed_mcp_sources"] = sources
            payload["fixed_read_case"] = read_case
            if args.mode in {"all", "finals"}:
                for repetition in range(1, args.repetitions + 1):
                    save(await measure(ToolLoop(), cases["S5"], "fixed_mcp_replay", repetition, sources))
                    agent = ToolLoop()
                    agent.last_trace = read_case[2]
                    save(await measure(agent, cases["S6"], "fixed_mcp_replay", repetition, read_case[0], read_case[1]))
            if args.mode in {"all", "unsupported"}:
                save(await measure(ToolLoop(), cases["S8"], "fixed_claim_real_verifier", sources=[sources[0]]))
        if args.mode in {"all", "autonomous"}:
            for repetition in range(1, args.repetitions + 1):
                for identifier in ("S5", "S6"):
                    save(await measure(ToolLoop(), cases[identifier], "autonomous", repetition))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "memory", "finals", "autonomous", "unsupported"), default="all")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(tempfile.gettempdir()) / "codeagent-m7-final.json")
    parser.add_argument("--restart-memory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("At least one repetition is required.")
    protocol = json.loads(DATASET.read_text(encoding="utf-8"))
    result = asyncio.run(run(args, protocol))
    if args.restart_memory:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
