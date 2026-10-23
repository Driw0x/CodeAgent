import argparse
import asyncio
import json
import os
import tempfile
import traceback
from pathlib import Path
from time import perf_counter

from app.agent import ToolLoop
from app.llm import LocalLLM
from app.memory import (
    build_index,
    build_manifest,
    compare_manifests,
    index_exists,
    load_index,
    load_manifest,
    manifest_exists,
    project_relative_path,
    save_index,
    save_manifest,
    update_index,
)
from app.parser import chunking, read_dir
from app.memory.context import ContextBudget
from app.memory.conversation import ConversationMemory
from app.memory.history import describe_error
from app.rag import RAGPipeline
from app.retrieval import FaissRetriever
from app.utils.paths import project_memory_dir

PROJECT_PATH = Path(__file__).resolve().parents[1]
MEMORY_ROOT = PROJECT_PATH / "data" / "memory"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
TOP_K = 5

AUTO_QUESTIONS = [
    "Quel est le point d'entrée principal du projet et que fait-il ?",
    "Comment les fichiers ou les données sont-ils chargés dans le projet ?",
    "Quelles sont les principales fonctions ou classes et à quoi servent-elles ?",
    "Comment les différents composants du projet interagissent-ils entre eux ?",
    "Quels mécanismes de gestion des erreurs sont présents dans le code ?",
]


def print_rag_stats(elapsed: float, stats: dict | None, title: str):
    print("\n=================================")
    print(title)
    print("=================================")
    if stats:
        print(f"Prompt tokens    : " f"{stats['prompt_tokens']}")
        print(f"Generated tokens : " f"{stats['completion_tokens']}")
        print(f"Total tokens     : " f"{stats['total_tokens']}")
    else:
        print("Token stats      : unavailable")
    print(f"Response time    : " f"{elapsed:.2f} s")


def print_tool_timings(agent: ToolLoop):
    print("\nTool timings")
    print("---------------------------------")

    if not agent.last_trace:
        print("No tool calls")
        return

    for index, trace in enumerate(agent.last_trace, start=1):
        arguments = json.dumps(trace.get("arguments", {}), ensure_ascii=False, separators=(",", ":"))
        print(
            f"{index}. "
            f"{trace.get('tool', 'unknown')} "
            f"[{trace.get('status', 'unknown')}] "
            f": "
            f"{trace.get('duration_seconds', 0.0):.2f} s"
        )
        print(f"   args={arguments}")


def print_agent_stats(agent: ToolLoop, elapsed: float):
    stats = agent.last_token_stats
    stages = agent.last_stage_stats
    times = agent.last_stage_times
    print("\n=================================")
    print("Agent question summary")
    print("=================================")
    print(f"Devstral prompt tokens     : " f"{stats['devstral_prompt']}")
    print(f"Devstral generated tokens  : " f"{stats['devstral_completion']}")
    print(f"Grounding prompt tokens     : " f"{stats['grounding_prompt']}")
    print(f"Grounding generated tokens  : " f"{stats['grounding_completion']}")
    print(f"Total tokens                : " f"{stats['total']}")
    if not stats["total"]:
        print("Token measurement           : unavailable (no completed model request)")
    print(f"Agent steps                 : " f"{agent.last_steps}")
    print(f"Tool calls                  : " f"{len(agent.last_trace)}")
    print(f"Grounding                   : " f"{agent.last_grounding}")
    execution = agent.last_execution
    print(f"Execution mode              : {execution.get('mode', 'unavailable')}")
    print(f"Stop reason                 : {execution.get('stop_reason') or 'M6 direct response'}")
    if execution.get("human"):
        print("HITL metrics                : " + json.dumps(execution["human"], ensure_ascii=False))
        print("Remaining budget            : " + json.dumps(execution["budgets"], ensure_ascii=False))
        print(f"Finalization attempts        : {len(execution.get('final_attempts', []))}")
    if execution.get("plan"):
        print(f"Completed / unfinished      : {execution['completed']} / {execution['remaining']}")
        print(f"Replanning count            : {execution['replans']}")
    print(f"Response time               : " f"{elapsed:.2f} s")
    print("\nStage details")
    print("---------------------------------")
    print(
        "Agent LLM tokens            : "
        f"{stages['agent']['prompt_tokens']} + "
        f"{stages['agent']['completion_tokens']}"
    )
    print(
        "Final answer tokens         : "
        f"{stages['final']['prompt_tokens']} + "
        f"{stages['final']['completion_tokens']}"
    )
    print(
        "Grounding tokens            : "
        f"{stages['grounding']['prompt_tokens']} + "
        f"{stages['grounding']['completion_tokens']}"
    )
    print(f"MCP setup time              : " f"{times['mcp_setup']:.2f} s")
    print(f"Planner tokens / time       : {stages['planner']['prompt_tokens']} + {stages['planner']['completion_tokens']} / {times['planner']:.2f} s")
    print(f"Agent LLM time              : " f"{times['agent']:.2f} s")
    print(f"Tools time                  : " f"{times['tools']:.2f} s")
    print(f"Final answer time           : " f"{times['final']:.2f} s")
    print(f"Grounding time              : " f"{times['grounding']:.2f} s")
    print_tool_timings(agent)


def ask_question(pipeline, llm, memory_dir, question: str, test_number: int | None = None):
    print(f"\nQuestion : {question}")
    llm.last_stats = None
    start = perf_counter()

    try:
        answer = pipeline.ask(question)
    except RuntimeError as error:
        print(f"\nError: {error}")
        return

    elapsed = perf_counter() - start
    print(f"\n{answer}")
    title = f"Test {test_number} summary" if test_number is not None else "Question summary"
    print_rag_stats(elapsed=elapsed, stats=llm.last_stats, title=title)
    print(f"Context: {getattr(pipeline, 'last_context_stats', {})}")


def run_manual(pipeline, llm, memory_dir):
    print("\nManual RAG mode")
    print("Press Enter without a question to quit.")
    while True:
        question = input("\nQuestion : ").strip()
        if not question:
            break
        ask_question(pipeline, llm, memory_dir, question)


def run_auto(pipeline, llm, memory_dir):
    print("\nAutomatic RAG test")
    print(f"{len(AUTO_QUESTIONS)} " "questions will be tested.")
    print("Warming up model...")
    llm.preload()
    for number, question in enumerate(AUTO_QUESTIONS, start=1):
        ask_question(pipeline, llm, memory_dir, question, test_number=number)


def preload_model(llm, label: str):
    print(f"Warming up " f"{label} ({llm.model})...")
    start = perf_counter()
    try:
        llm.preload()
    except RuntimeError as error:
        print(f"{label} warm-up failed: " f"{error}")
        return
    elapsed = perf_counter() - start
    print(f"{label} ready in " f"{elapsed:.2f} s")


def run_agent(session_id="default", context_tokens=None, **execution_options):
    print("\nAgent mode")
    llm = LocalLLM(model="devstral:24b", context_tokens=context_tokens)
    memory_dir = project_memory_dir(PROJECT_PATH, MEMORY_ROOT)
    memory = ConversationMemory(memory_dir, session_id, max_tokens=max(512, ContextBudget.for_llm(llm).input_limit // 3))
    agent = ToolLoop(llm=llm, memory=memory, **execution_options)
    if context_tokens is not None:
        agent.grounding_llm = LocalLLM(model=agent.grounding_llm.model, context_tokens=context_tokens)
    print(f"Session: {session_id} | checkpoint: {memory.state_path}")
    print(f"Tool model      : " f"{agent.llm.model}")
    print(f"Grounding model : " f"{agent.grounding_llm.model}")

    if agent.grounding_llm.model != agent.llm.model:
        preload_model(agent.grounding_llm, "Grounding model")

    preload_model(agent.llm, "Tool model")
    print("Press Enter without a question to quit.")

    while True:
        question = input("\nQuestion : ").strip()

        if not question:
            break

        start = perf_counter()

        try:
            answer = asyncio.run(agent.ask(question))
        except Exception as error:
            elapsed = perf_counter() - start
            print(f"\nError: {describe_error(error)}")
            if agent.debug:
                traceback.print_exception(error)
                if agent.last_debug:
                    print("Debug: " + json.dumps(agent.last_debug, ensure_ascii=False))

            print_agent_stats(agent, elapsed)
            continue

        elapsed = perf_counter() - start
        print(f"\n{answer}")
        print_agent_stats(agent, elapsed)
        print(f"Context: {getattr(memory, 'last_stats', {})}")
        if agent.debug and agent.last_debug:
            print("Debug: " + json.dumps(agent.last_debug, ensure_ascii=False))


def _save_rag_state(index, chunks, manifest, directory):
    """Stage all files before publishing; a missing manifest invalidates interrupted writes."""
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rag-update-", dir=directory) as temporary:
        staged = Path(temporary)
        save_index(index, chunks, staged)
        save_manifest(manifest, staged)
        # The old snapshot survives build/serialization failures. Once publication starts,
        # remove its validity marker before replacing either member of the index pair.
        (directory / "manifest.json").unlink(missing_ok=True)
        for name in ("index.faiss", "chunks.json", "manifest.json"):
            (staged / name).replace(directory / name)


def build_rag_pipeline(session_id="default", context_tokens=None):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    memory_dir = project_memory_dir(PROJECT_PATH, MEMORY_ROOT)
    files = read_dir(PROJECT_PATH)
    current_manifest = build_manifest(files, PROJECT_PATH)

    if index_exists(memory_dir) and manifest_exists(memory_dir):
        print("Loading index from memory...")
        index, chunks = load_index(memory_dir)

        previous_manifest = load_manifest(memory_dir)
        changes = compare_manifests(previous_manifest, current_manifest)

        if changes["added"] or changes["modified"] or changes["deleted"]:
            print("\nProject changes detected:")

            for path in changes["added"]:
                print(f"  + Added:    {path}")

            for path in changes["modified"]:
                print(f"  ~ Modified: {path}")

            for path in changes["deleted"]:
                print(f"  - Deleted:  {path}")

            changed_paths = changes["added"] + changes["modified"] + changes["deleted"]
            files_by_path = {
                project_relative_path(file["path"], PROJECT_PATH): file
                for file in files
            }
            new_chunks = []

            for path in (changes["added"] + changes["modified"]):
                new_chunks.extend(chunking(files_by_path[path]))

            index, chunks = (
                update_index(
                    model=model,
                    index=index,
                    chunks=chunks,
                    new_chunks=new_chunks,
                    changed_paths=changed_paths,
                    project_path=PROJECT_PATH,
                )
            )
            _save_rag_state(index, chunks, current_manifest, memory_dir)
            print(
                f"\nIndex updated: "
                f"{len(changes['added'])} added, "
                f"{len(changes['modified'])} modified, "
                f"{len(changes['deleted'])} deleted."
            )
        else:
            print("No project changes detected.")
    else:
        print("Building index...")
        chunks = []

        for file in files:
            chunks.extend(chunking(file))

        index = build_index(model=model, chunks=chunks, dimension=EMBEDDING_DIMENSION)
        _save_rag_state(index, chunks, current_manifest, memory_dir)

    retriever = FaissRetriever(model=model, index=index, chunks=chunks)
    llm = LocalLLM(context_tokens=context_tokens)
    memory = ConversationMemory(memory_dir, session_id, max_tokens=max(512, ContextBudget.for_llm(llm).input_limit // 3))
    pipeline = RAGPipeline(retriever=retriever.retrieve, llm=llm, top_k=TOP_K, memory=memory)
    print(f"Session: {session_id} | checkpoint: {memory.state_path}")

    return pipeline, llm, memory_dir


def main():
    global PROJECT_PATH
    parser = argparse.ArgumentParser(description="CodeAgent: local RAG and MCP conversations.")
    parser.add_argument("--session", default="default", help="Persistent session ID; choose a new ID for a separate conversation.")
    parser.add_argument("--project", type=Path, default=PROJECT_PATH, help="Python project to inspect.")
    parser.add_argument("--context-tokens", type=int, help="Ollama num_ctx and shared context budget.")
    parser.add_argument("--debug", action="store_true", help="Agent diagnostics: full exception traceback and grounding candidate/sources in terminal and local history.")
    parser.add_argument("--planning", choices=("auto", "off", "adaptive"), default="auto", help="M8 global planning, direct M6, or experimental adaptive execution.")
    parser.add_argument("--hitl", choices=("off", "on-failure"), default="off", help="Bounded human guidance on M8 failures; requires interactive stdin.")
    parser.add_argument("--max-steps", type=int, default=12, help="Global M6 iterations per question (including all planned steps).")
    parser.add_argument("--max-plan-steps", type=int, default=6)
    parser.add_argument("--max-replans", type=int, default=2)
    parser.add_argument("--max-tool-calls", type=int, default=16)
    parser.add_argument("--max-total-tokens", type=int, default=32768, help="Global LLM token ceiling; conservative reservation before each request.")
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--tool-timeout", type=float, default=60)
    args = parser.parse_args()
    if args.debug:
        os.environ["CODEAGENT_DEBUG"] = "1"
    PROJECT_PATH = args.project.resolve()
    if not PROJECT_PATH.is_dir():
        parser.error("Project must be an existing directory.")
    os.environ["CODEAGENT_PROJECT_PATH"] = str(PROJECT_PATH)
    print("\nChoose mode:")
    print("[0] - Manual RAG")
    print("[1] - Automatic RAG test")
    print("[2] - Agent")
    mode = input("\nChoice : ").strip()

    if mode == "2":
        run_agent(args.session, args.context_tokens, planning=args.planning, hitl=args.hitl, max_steps=args.max_steps,
                  max_plan_steps=args.max_plan_steps, max_replans=args.max_replans,
                  max_tool_calls=args.max_tool_calls, max_total_tokens=args.max_total_tokens,
                  max_seconds=args.max_seconds, tool_timeout=args.tool_timeout)
        return

    pipeline, llm, memory_dir = build_rag_pipeline(args.session, args.context_tokens)

    if mode == "1":
        run_auto(pipeline, llm, memory_dir)
    else:
        run_manual(pipeline, llm, memory_dir)


if __name__ == "__main__":
    main()
