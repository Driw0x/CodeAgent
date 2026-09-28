import json
import time
from pathlib import Path

from app.llm import LocalLLM
from app.rag.citations import extract_citations, is_abstention, validate_citations
from app.rag.grounding import verify_grounding
from app.rag.prompt import SYSTEM_PROMPT, build_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "benchmarks" / "rag" / "fixture" / "chunks.json"
QUESTIONS = PROJECT_ROOT / "benchmarks" / "rag" / "grounding_questions.json"
OUTPUT = PROJECT_ROOT / "benchmarks" / "rag" / "results" / "m5_grounding_verified.json"


def relative_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def find_chunks(chunks: list[dict], sources: list[dict]) -> list[dict]:
    result = []

    for source in sources:
        for chunk in chunks:
            if relative_path(chunk["file"]) == source["file"] and chunk.get("type") == source["type"] and chunk.get("name") == source["name"]:
                result.append(chunk)
                break

    return result


def evaluate(item: dict, chunks: list[dict], llm: LocalLLM) -> dict:
    context_chunks = find_chunks(chunks, item["sources"])

    if len(context_chunks) != len(item["sources"]):
        raise ValueError(f"{item['id']}: source missing from fixture")

    prompt = build_prompt(item["question"], context_chunks)

    start = time.perf_counter()
    answer = llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
    response_time = time.perf_counter() - start

    found_citations = extract_citations(answer)
    has_citations = bool(found_citations)
    valid_citations = validate_citations(answer, len(context_chunks))
    abstention = is_abstention(answer)

    if item["category"] == "answerable":
        success = has_citations and valid_citations and not abstention
    elif item["category"] == "partial":
        success = has_citations and valid_citations and abstention
    elif item["category"] == "unanswerable":
        success = abstention and valid_citations
    else:
        raise ValueError(f"{item['id']}: invalid category")

    grounding_checked = has_citations and valid_citations
    grounding_time = 0.0
    grounding_supported = None

    if grounding_checked:
        start = time.perf_counter()
        grounding_supported = verify_grounding(answer, context_chunks, llm)
        grounding_time = time.perf_counter() - start

    verification_match = grounding_supported == success if grounding_checked else success

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "answer": answer,
        "citations": found_citations,
        "has_citations": has_citations,
        "valid_citations": valid_citations,
        "abstention": abstention,
        "success": success,
        "grounding_checked": grounding_checked,
        "grounding_supported": grounding_supported,
        "verification_match": verification_match,
        "response_time": response_time,
        "grounding_time": grounding_time,
    }


def main() -> None:
    chunks = json.loads(FIXTURE.read_text(encoding="utf-8"))
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    llm = LocalLLM()

    results = []
    total_start = time.perf_counter()

    for item in questions:
        result = evaluate(item, chunks, llm)
        results.append(result)

        grounding = "SUPPORTED" if result["grounding_supported"] is True else "UNSUPPORTED" if result["grounding_supported"] is False else "SKIPPED"

        print(
            f"{result['id']} | {result['category']} | "
            f"success={result['success']} | "
            f"grounding={grounding} | "
            f"match={result['verification_match']} | "
            f"generation={result['response_time']:.2f}s | "
            f"verification={result['grounding_time']:.2f}s"
        )
        print(f"Answer: {result['answer']}\n")

    total_time = time.perf_counter() - total_start
    success_count = sum(result["success"] for result in results)
    grounding_checked = sum(result["grounding_checked"] for result in results)
    grounding_supported = sum(result["grounding_supported"] is True for result in results)
    grounding_rejected = sum(result["grounding_supported"] is False for result in results)
    verification_matches = sum(result["verification_match"] for result in results)
    missing_citations = sum(result["category"] in {"answerable", "partial"} and not result["has_citations"] for result in results)
    invalid_citations = sum(bool(result["citations"]) and not result["valid_citations"] for result in results)
    average_response_time = sum(result["response_time"] for result in results) / len(results)
    average_grounding_time = sum(result["grounding_time"] for result in results if result["grounding_checked"]) / grounding_checked if grounding_checked else 0.0

    summary = {
        "questions": len(results),
        "prompt_success": success_count,
        "prompt_success_rate": success_count / len(results),
        "grounding_checked": grounding_checked,
        "grounding_supported": grounding_supported,
        "grounding_rejected": grounding_rejected,
        "verification_matches": verification_matches,
        "verification_match_rate": verification_matches / len(results),
        "missing_citations": missing_citations,
        "invalid_citations": invalid_citations,
        "average_response_time": average_response_time,
        "average_grounding_time": average_grounding_time,
        "total_time": total_time,
    }

    print("========== GROUNDING VERIFICATION BENCHMARK ==========")
    print(f"Questions                 : {summary['questions']}")
    print(f"Prompt success            : {summary['prompt_success']}")
    print(f"Prompt success rate       : {summary['prompt_success_rate']:.4f}")
    print(f"Grounding checked         : {summary['grounding_checked']}")
    print(f"Grounding supported       : {summary['grounding_supported']}")
    print(f"Grounding rejected        : {summary['grounding_rejected']}")
    print(f"Verification matches      : {summary['verification_matches']}")
    print(f"Verification match rate   : {summary['verification_match_rate']:.4f}")
    print(f"Missing citations         : {summary['missing_citations']}")
    print(f"Invalid citations         : {summary['invalid_citations']}")
    print(f"Average generation time   : {summary['average_response_time']:.2f}s")
    print(f"Average verification time : {summary['average_grounding_time']:.2f}s")
    print(f"Total time                : {summary['total_time']:.2f}s")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nResults saved to: {OUTPUT}")


if __name__ == "__main__":
    main()