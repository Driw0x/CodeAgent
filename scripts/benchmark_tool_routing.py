import asyncio
import json
from pathlib import Path
from time import perf_counter

from mcp import Client

from app.agent.tool_loop import mcp_tool_to_ollama
from app.llm import LocalLLM
from app.mcp_server import mcp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "benchmarks" / "tools" / "baseline_tasks.json"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "tools" / "results"
RESULT_PATH = RESULTS_DIR / "devstral_routing_baseline.json"
MODEL = "devstral:24b"


def expected_tools(task: dict) -> set[str]:
    if "expected_tools" in task:
        return set(task["expected_tools"])
    tool = task.get("expected_tool")
    return {tool} if tool is not None else set()


def arguments_match(actual: dict, expected: dict) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


async def load_tools() -> list[dict]:
    async with Client(mcp) as client:
        listed = await client.list_tools()
    return [mcp_tool_to_ollama(tool) for tool in listed.tools]


def warm_up(llm: LocalLLM) -> None:
    print("Warming up Devstral...")
    llm.chat(messages=[{"role": "user", "content": "Réponds uniquement par OK."}], tools=None)
    print("Warm-up complete.\n")


def run_task(llm: LocalLLM, tools: list[dict], task: dict) -> dict:
    start = perf_counter()
    error = None

    try:
        message = llm.chat(messages=[{"role": "user", "content": task["prompt"]}], tools=tools)
    except Exception as exc:
        message = {}
        error = str(exc)

    latency = perf_counter() - start
    tool_calls = message.get("tool_calls") or []
    first_call = tool_calls[0] if tool_calls else None
    function = first_call.get("function", {}) if first_call else {}
    first_tool = function.get("name")
    arguments = function.get("arguments", {})

    if not isinstance(arguments, dict):
        arguments = {}

    valid_tools = expected_tools(task)
    tool_selection_correct = first_tool in valid_tools if valid_tools else first_tool is None
    expected_arguments = task.get("expected_arguments", {})

    if not valid_tools:
        arguments_correct = first_tool is None
    elif first_tool in valid_tools:
        arguments_correct = arguments_match(arguments, expected_arguments)
    else:
        arguments_correct = False

    routing_success = error is None and tool_selection_correct and arguments_correct

    return {
        "id": task["id"],
        "prompt": task["prompt"],
        "expected_tools": sorted(valid_tools),
        "expected_arguments": expected_arguments,
        "first_tool": first_tool,
        "arguments": arguments,
        "tool_selection_correct": tool_selection_correct,
        "arguments_correct": arguments_correct,
        "routing_success": routing_success,
        "latency_seconds": round(latency, 4),
        "content": message.get("content", ""),
        "error": error,
    }


def summarize(results: list[dict]) -> dict:
    count = len(results)
    tool_tasks = [result for result in results if result["expected_tools"]]
    no_tool_tasks = [result for result in results if not result["expected_tools"]]
    successful_latencies = [
        result["latency_seconds"]
        for result in results
        if result["error"] is None
    ]

    return {
        "tasks": count,
        "routing_success_rate": (
            sum(result["routing_success"] for result in results) / count
        ),
        "tool_selection_accuracy": (
            sum(result["tool_selection_correct"] for result in tool_tasks)
            / len(tool_tasks)
            if tool_tasks
            else 0.0
        ),
        "no_tool_accuracy": (
            sum(result["first_tool"] is None for result in no_tool_tasks)
            / len(no_tool_tasks)
            if no_tool_tasks
            else 0.0
        ),
        "argument_accuracy": (
            sum(result["arguments_correct"] for result in tool_tasks)
            / len(tool_tasks)
            if tool_tasks
            else 0.0
        ),
        "average_routing_latency_seconds": (
            sum(successful_latencies) / len(successful_latencies)
            if successful_latencies
            else 0.0
        ),
        "min_routing_latency_seconds": (
            min(successful_latencies)
            if successful_latencies
            else 0.0
        ),
        "max_routing_latency_seconds": (
            max(successful_latencies)
            if successful_latencies
            else 0.0
        ),
        "errors": sum(result["error"] is not None for result in results),
    }


async def main():
    tasks = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    tools = await load_tools()
    llm = LocalLLM(model=MODEL)
    warm_up(llm)
    results = []

    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {task['id']}")
        result = run_task(llm, tools, task)
        results.append(result)
        print(
            f"  success={result['routing_success']} "
            f"tool={result['first_tool']} "
            f"arguments={result['arguments_correct']} "
            f"latency={result['latency_seconds']:.2f}s"
        )

    summary = summarize(results)
    output = {
        "model": MODEL,
        "dataset": DATASET_PATH.name,
        "benchmark": "tool_routing",
        "summary": summary,
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSummary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to {RESULT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
