import asyncio
import json
from pathlib import Path
from time import perf_counter

from app.agent import ToolLoop

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "benchmarks" / "tools" / "baseline_tasks.json"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "tools" / "results"
RESULT_PATH = RESULTS_DIR / "devstral_baseline.json"
MODEL = "devstral:24b"


def expected_tools(task: dict) -> set[str]:
    if "expected_tools" in task:
        return set(task["expected_tools"])
    tool = task.get("expected_tool")
    return {tool} if tool is not None else set()


def arguments_match(actual: dict, expected: dict) -> bool:
    return all(actual.get(key) == value for key, value in expected.items())


def answer_matches(answer: str, expected: list[str]) -> bool:
    return all(value.casefold() in answer.casefold() for value in expected)


def warm_up(agent: ToolLoop) -> None:
    print("Warming up grounding model...")
    agent.grounding_llm.generate("Réponds uniquement par OK.")
    print("Warming up Devstral...")
    agent.llm.chat(messages=[{"role": "user", "content": "Réponds uniquement par OK."}], tools=None)
    print("Warm-up complete.\n")


async def run_task(agent: ToolLoop, task: dict) -> dict:
    start = perf_counter()
    error = None

    try:
        answer = await agent.ask(task["prompt"])
    except Exception as exc:
        answer = ""
        error = str(exc)

    latency = perf_counter() - start
    trace = list(agent.last_trace)
    first_tool = trace[0]["tool"] if trace else None
    valid_tools = expected_tools(task)
    tool_selection_correct = first_tool in valid_tools if valid_tools else first_tool is None
    expected_arguments = task.get("expected_arguments", {})

    if not valid_tools:
        arguments_correct = len(trace) == 0
    elif trace:
        arguments_correct = arguments_match(trace[0].get("arguments", {}), expected_arguments)
    else:
        arguments_correct = False

    expected_answer = task.get("expected_answer_contains", [])
    answer_correct = answer_matches(answer, expected_answer) if expected_answer else True
    routing_success = tool_selection_correct and arguments_correct

    if valid_tools:
        grounding_success = agent.last_grounding == "passed"
    else:
        grounding_success = agent.last_grounding == "not_used"

    end_to_end_success = error is None and routing_success and answer_correct and grounding_success

    return {
        "id": task["id"],
        "prompt": task["prompt"],
        "expected_tools": sorted(valid_tools),
        "first_tool": first_tool,
        "tool_selection_correct": tool_selection_correct,
        "arguments_correct": arguments_correct,
        "routing_success": routing_success,
        "answer_correct": answer_correct,
        "grounding_success": grounding_success,
        "task_success": end_to_end_success,
        "steps": agent.last_steps,
        "tool_calls": len(trace),
        "tool_errors": sum(
            item["status"] in {"error", "exception", "timeout"}
            for item in trace
        ),
        "duplicates": sum(item["status"] == "duplicate" for item in trace),
        "retry_limits": sum(item["status"] == "retry_limit" for item in trace),
        "grounding": agent.last_grounding,
        "latency_seconds": round(latency, 4),
        "answer": answer,
        "error": error,
        "trace": trace,
    }


def summarize(results: list[dict]) -> dict:
    count = len(results)
    tool_tasks = [result for result in results if result["expected_tools"]]
    no_tool_tasks = [result for result in results if not result["expected_tools"]]

    return {
        "tasks": count,
        "routing_success_rate": (
            sum(result["routing_success"] for result in results) / count
        ),
        "end_to_end_success_rate": (
            sum(result["task_success"] for result in results) / count
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
        "average_steps": (
            sum(result["steps"] for result in results) / count
        ),
        "average_tool_calls": (
            sum(result["tool_calls"] for result in results) / count
        ),
        "average_latency_seconds": (
            sum(result["latency_seconds"] for result in results) / count
        ),
        "tool_errors": sum(result["tool_errors"] for result in results),
        "duplicates": sum(result["duplicates"] for result in results),
        "retry_limits": sum(result["retry_limits"] for result in results),
        "grounding": {
            status: sum(result["grounding"] == status for result in results)
            for status in {
                "not_used",
                "passed",
                "citation_error",
                "unsupported",
            }
        },
    }


async def main():
    tasks = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    agent = ToolLoop()
    warm_up(agent)
    results = []

    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {task['id']}")
        result = await run_task(agent, task)
        results.append(result)
        print(
            f"  routing={result['routing_success']} "
            f"e2e={result['task_success']} "
            f"tool={result['first_tool']} "
            f"grounding={result['grounding']} "
            f"steps={result['steps']} "
            f"latency={result['latency_seconds']:.2f}s"
        )

    summary = summarize(results)
    output = {"model": MODEL, "dataset": DATASET_PATH.name, "summary": summary, "results": results}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSummary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to {RESULT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
