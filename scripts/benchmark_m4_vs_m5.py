import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.llm import LocalLLM
from app.memory.vector_store import load_index
from app.rag.citations import (
    extract_citations,
    is_abstention,
    validate_citations,
)
from app.rag.grounding import (
    is_missing_information,
    verify_grounding,
)
from app.rag.prompt import (
    SYSTEM_PROMPT,
    build_prompt,
    select_context_chunks,
)
from app.retrieval.faiss_retriever import (
    FaissRetriever,
    STOPWORDS,
    TYPE_BONUS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIXTURE = (
    PROJECT_ROOT
    / "benchmarks"
    / "rag"
    / "fixture"
)

QUESTIONS = (
    PROJECT_ROOT
    / "benchmarks"
    / "rag"
    / "final_questions.json"
)

RESULTS = (
    PROJECT_ROOT
    / "benchmarks"
    / "rag"
    / "results"
)

M4_OUTPUT = RESULTS / "m4_final.json"
M5_OUTPUT = RESULTS / "m5_final.json"
SUMMARY_OUTPUT = RESULTS / "m4_vs_m5_summary.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
M4_CANDIDATES = 25


M4_SYSTEM_PROMPT = """Tu es un assistant spécialisé dans l'analyse de code source.

Réponds uniquement à partir du contexte fourni.
Si l'information demandée n'est pas présente dans le contexte, indique clairement que tu ne peux pas la déterminer.
N'invente jamais de fonction, fichier, comportement ou dépendance.
Lorsque tu utilises un morceau de code, référence son fichier et ses lignes.
"""


def relative_path(path: str | Path) -> str:
    path = Path(path)

    try:
        return path.resolve().relative_to(
            PROJECT_ROOT
        ).as_posix()

    except ValueError:
        return path.as_posix()


def source_matches(
    chunk: dict,
    expected: dict,
) -> bool:
    return (
        relative_path(chunk["file"])
        == expected["file"]
        and chunk.get("type")
        == expected.get("type")
        and chunk.get("name")
        == expected.get("name")
    )


def retrieval_metrics(
    chunks: list[dict],
    expected_sources: list[dict],
) -> dict | None:
    if not expected_sources:
        return None

    result = {}

    for k in (1, 3, 5):
        top = chunks[:k]

        matched = {
            i
            for i, expected
            in enumerate(expected_sources)
            if any(
                source_matches(
                    chunk,
                    expected,
                )
                for chunk in top
            )
        }

        relevant = sum(
            any(
                source_matches(
                    chunk,
                    expected,
                )
                for expected in expected_sources
            )
            for chunk in top
        )

        result[f"hit@{k}"] = (
            1.0 if matched else 0.0
        )

        result[f"recall@{k}"] = (
            len(matched)
            / len(expected_sources)
        )

        result[f"precision@{k}"] = (
            relevant / k
        )

    result["mrr@5"] = 0.0

    for rank, chunk in enumerate(
        chunks[:5],
        start=1,
    ):
        if any(
            source_matches(
                chunk,
                expected,
            )
            for expected in expected_sources
        ):
            result["mrr@5"] = 1.0 / rank
            break

    return result


def serialize_chunks(
    chunks: list[dict],
) -> list[dict]:
    return [
        {
            "rank": rank,
            "file": relative_path(
                chunk["file"]
            ),
            "type": chunk.get("type"),
            "name": chunk.get("name"),
        }
        for rank, chunk
        in enumerate(
            chunks,
            start=1,
        )
    ]


# ------------------------------------------------------------------
# M4 retrieval
# ------------------------------------------------------------------

def m4_tokenize(
    text: str,
) -> set[str]:
    normalized = unicodedata.normalize(
        "NFKD",
        text.casefold(),
    )

    normalized = "".join(
        char
        for char in normalized
        if not unicodedata.combining(char)
    )

    return {
        token
        for token in re.findall(
            r"[a-z0-9_]+",
            normalized,
        )
        if token not in STOPWORDS
    }


def m4_rerank_score(
    question: str,
    chunk: dict,
) -> float:
    question_tokens = m4_tokenize(
        question
    )

    if not question_tokens:
        return 0.0

    name_tokens = m4_tokenize(
        str(
            chunk.get(
                "name",
                "",
            )
        )
    )

    file_tokens = m4_tokenize(
        relative_path(
            chunk.get(
                "file",
                "",
            )
        )
    )

    content_tokens = m4_tokenize(
        str(
            chunk.get(
                "content",
                "",
            )
        )
    )

    all_tokens = (
        name_tokens
        | file_tokens
        | content_tokens
    )

    coverage = (
        len(
            question_tokens
            & all_tokens
        )
        / len(question_tokens)
    )

    name_overlap = len(
        question_tokens
        & name_tokens
    )

    file_overlap = len(
        question_tokens
        & file_tokens
    )

    type_bonus = TYPE_BONUS.get(
        chunk.get("type"),
        0.0,
    )

    return (
        10.0 * coverage
        + 2.0 * name_overlap
        + file_overlap
        + type_bonus
    )


def m4_dense_retrieve(
    retriever: FaissRetriever,
    question: str,
    k: int,
) -> list[dict]:
    vector = retriever.model.encode(
        [question],
        show_progress_bar=False,
    )

    vector = np.asarray(
        vector,
        dtype="float32",
    )

    k = min(
        k,
        len(retriever.chunks),
    )

    distances, indices = (
        retriever.index.search(
            vector,
            k,
        )
    )

    results = []

    for distance, index in zip(
        distances[0],
        indices[0],
    ):
        if (
            index < 0
            or index
            >= len(retriever.chunks)
        ):
            continue

        chunk = dict(
            retriever.chunks[index]
        )

        chunk["distance"] = float(
            distance
        )

        results.append(chunk)

    return results


def m4_retrieve(
    retriever: FaissRetriever,
    question: str,
) -> list[dict]:
    candidates = m4_dense_retrieve(
        retriever,
        question,
        M4_CANDIDATES,
    )

    for chunk in candidates:
        chunk["rerank_score"] = (
            m4_rerank_score(
                question,
                chunk,
            )
        )

    candidates.sort(
        key=lambda chunk: (
            -chunk["rerank_score"],
            chunk["distance"],
        )
    )

    return candidates[:TOP_K]


def m4_context(
    chunks: list[dict],
) -> str:
    return "\n\n---\n\n".join(
        (
            f"File: "
            f"{relative_path(chunk['file'])}\n"
            f"Lines: "
            f"{chunk['start_line']}-"
            f"{chunk['end_line']}\n"
            f"Type: {chunk['type']}\n"
            f"Name: {chunk['name']}\n\n"
            f"{chunk['content']}"
        )
        for chunk in chunks
    )


def m4_prompt(
    question: str,
    chunks: list[dict],
) -> str:
    return (
        f"CONTEXTE:\n"
        f"{m4_context(chunks)}"
        f"\n\nQUESTION:\n"
        f"{question}"
    )


# ------------------------------------------------------------------
# Response behaviour
# ------------------------------------------------------------------

def has_answer_content(
    answer: str,
) -> bool:
    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer,
    )

    for sentence in sentences:
        sentence = sentence.strip()

        if (
            sentence
            and not is_abstention(
                sentence
            )
            and not is_missing_information(
                sentence
            )
        ):
            return True

    return False


def behavior_match(
    category: str,
    answer: str,
) -> bool:
    abstention = is_abstention(
        answer
    )

    if category == "answerable":
        return not abstention

    if category == "partial":
        return (
            abstention
            and has_answer_content(
                answer
            )
        )

    if category == "unanswerable":
        return abstention

    raise ValueError(
        f"Invalid category: "
        f"{category}"
    )


# ------------------------------------------------------------------
# M4 evaluation
# ------------------------------------------------------------------

def evaluate_m4(
    item: dict,
    retriever: FaissRetriever,
    llm: LocalLLM,
) -> dict:
    chunks = m4_retrieve(
        retriever,
        item["question"],
    )

    metrics = retrieval_metrics(
        chunks,
        item["expected_sources"],
    )

    start = time.perf_counter()

    answer = llm.generate(
        prompt=m4_prompt(
            item["question"],
            chunks,
        ),
        system_prompt=M4_SYSTEM_PROMPT,
    )

    generation_time = (
        time.perf_counter()
        - start
    )

    response_behavior_match = (
        behavior_match(
            item["category"],
            answer,
        )
    )

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "retrieved_sources": (
            serialize_chunks(chunks)
        ),
        "retrieval": metrics,
        "answer": answer,
        "abstention": (
            is_abstention(answer)
        ),
        "has_answer_content": (
            has_answer_content(answer)
        ),
        "response_behavior_match": (
            response_behavior_match
        ),
        "generation_time": (
            generation_time
        ),
    }


# ------------------------------------------------------------------
# M5 evaluation
# ------------------------------------------------------------------

def evaluate_m5(
    item: dict,
    retriever: FaissRetriever,
    llm: LocalLLM,
) -> dict:
    retrieved = retriever.retrieve(
        item["question"],
        k=TOP_K,
    )

    metrics = retrieval_metrics(
        retrieved,
        item["expected_sources"],
    )

    context_chunks = (
        select_context_chunks(
            retrieved
        )
    )

    start = time.perf_counter()

    answer = llm.generate(
        prompt=build_prompt(
            item["question"],
            context_chunks,
        ),
        system_prompt=SYSTEM_PROMPT,
    )

    generation_time = (
        time.perf_counter()
        - start
    )

    citations = extract_citations(
        answer
    )

    has_citations = bool(
        citations
    )

    if has_citations:
        citations_in_range = all(
            1
            <= citation
            <= len(context_chunks)
            for citation in citations
        )
    else:
        citations_in_range = None

    citation_validation_passed = (
        validate_citations(
            answer,
            len(context_chunks),
        )
    )

    abstention = is_abstention(
        answer
    )

    answer_has_content = (
        has_answer_content(
            answer
        )
    )

    grounding_checked = (
        citations_in_range is True
    )

    grounding_supported = None
    grounding_time = 0.0

    if grounding_checked:
        start = time.perf_counter()

        grounding_supported = (
            verify_grounding(
                answer,
                context_chunks,
                llm,
            )
        )

        grounding_time = (
            time.perf_counter()
            - start
        )

    # A technical/factual answer must:
    # - contain citations
    # - reference valid source ids
    # - pass citation validation
    # - pass semantic grounding
    #
    # A complete abstention may be accepted
    # without citations.
    if answer_has_content:
        pipeline_accepted = (
            has_citations
            and citations_in_range
            is True
            and citation_validation_passed
            and grounding_supported
            is True
        )
    else:
        pipeline_accepted = (
            citation_validation_passed
        )

    response_behavior_match = (
        behavior_match(
            item["category"],
            answer,
        )
    )

    final_acceptance_match = (
        response_behavior_match
        and pipeline_accepted
    )

    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "retrieved_sources": (
            serialize_chunks(
                retrieved
            )
        ),
        "retrieval": metrics,
        "answer": answer,
        "citations": citations,
        "has_citations": (
            has_citations
        ),
        "citations_in_range": (
            citations_in_range
        ),
        "citation_validation_passed": (
            citation_validation_passed
        ),
        "abstention": abstention,
        "has_answer_content": (
            answer_has_content
        ),
        "grounding_checked": (
            grounding_checked
        ),
        "grounding_supported": (
            grounding_supported
        ),
        "pipeline_accepted": (
            pipeline_accepted
        ),
        "response_behavior_match": (
            response_behavior_match
        ),
        "final_acceptance_match": (
            final_acceptance_match
        ),
        "generation_time": (
            generation_time
        ),
        "grounding_time": (
            grounding_time
        ),
        "total_time": (
            generation_time
            + grounding_time
        ),
    }


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------

def mean(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def aggregate_retrieval(
    results: list[dict],
) -> dict:
    rows = [
        result["retrieval"]
        for result in results
        if result["retrieval"]
        is not None
    ]

    if not rows:
        return {}

    return {
        key: mean(
            [
                row[key]
                for row in rows
            ]
        )
        for key in rows[0]
    }


def aggregate_m4(
    results: list[dict],
) -> dict:
    return {
        "questions": len(results),
        "retrieval_questions": sum(
            result["retrieval"]
            is not None
            for result in results
        ),
        "retrieval": (
            aggregate_retrieval(
                results
            )
        ),
        "response_behavior_matches": sum(
            result[
                "response_behavior_match"
            ]
            for result in results
        ),
        "response_behavior_match_rate": mean(
            [
                float(
                    result[
                        "response_behavior_match"
                    ]
                )
                for result in results
            ]
        ),
        "average_generation_time": mean(
            [
                result[
                    "generation_time"
                ]
                for result in results
            ]
        ),
    }


def aggregate_m5(
    results: list[dict],
) -> dict:
    grounding = [
        result
        for result in results
        if result[
            "grounding_checked"
        ]
    ]

    citation_expected = [
        result
        for result in results
        if result["category"]
        in {
            "answerable",
            "partial",
        }
    ]

    return {
        "questions": len(results),

        "retrieval_questions": sum(
            result["retrieval"]
            is not None
            for result in results
        ),

        "retrieval": (
            aggregate_retrieval(
                results
            )
        ),

        "response_behavior_matches": sum(
            result[
                "response_behavior_match"
            ]
            for result in results
        ),

        "response_behavior_match_rate": mean(
            [
                float(
                    result[
                        "response_behavior_match"
                    ]
                )
                for result in results
            ]
        ),

        "final_acceptance_matches": sum(
            result[
                "final_acceptance_match"
            ]
            for result in results
        ),

        "final_acceptance_match_rate": mean(
            [
                float(
                    result[
                        "final_acceptance_match"
                    ]
                )
                for result in results
            ]
        ),

        "citation_presence_rate": mean(
            [
                float(
                    result[
                        "has_citations"
                    ]
                )
                for result
                in citation_expected
            ]
        ),

        "invalid_citation_count": sum(
            result["has_citations"]
            and result[
                "citations_in_range"
            ]
            is False
            for result in results
        ),

        "citation_validation_pass_rate": mean(
            [
                float(
                    result[
                        "citation_validation_passed"
                    ]
                )
                for result in results
            ]
        ),

        "pipeline_accept_rate": mean(
            [
                float(
                    result[
                        "pipeline_accepted"
                    ]
                )
                for result in results
            ]
        ),

        "grounding_checked": len(
            grounding
        ),

        "grounding_supported": sum(
            result[
                "grounding_supported"
            ]
            is True
            for result in grounding
        ),

        "grounding_rejected": sum(
            result[
                "grounding_supported"
            ]
            is False
            for result in grounding
        ),

        "average_generation_time": mean(
            [
                result[
                    "generation_time"
                ]
                for result in results
            ]
        ),

        "average_grounding_time": mean(
            [
                result[
                    "grounding_time"
                ]
                for result in grounding
            ]
        ),

        "average_total_time": mean(
            [
                result[
                    "total_time"
                ]
                for result in results
            ]
        ),
    }


# ------------------------------------------------------------------
# Comparison
# ------------------------------------------------------------------

def build_comparison(
    m4: dict,
    m5: dict,
) -> dict:
    return {
        "m4": m4,
        "m5": m5,
        "delta": {
            "retrieval": {
                key: (
                    m5["retrieval"][key]
                    - m4["retrieval"][key]
                )
                for key
                in m4["retrieval"]
            },

            "response_behavior_match_rate": (
                m5[
                    "response_behavior_match_rate"
                ]
                - m4[
                    "response_behavior_match_rate"
                ]
            ),

            "average_total_time": (
                m5[
                    "average_total_time"
                ]
                - m4[
                    "average_generation_time"
                ]
            ),
        },
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    questions = json.loads(
        QUESTIONS.read_text(
            encoding="utf-8"
        )
    )

    print(
        "Loading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Loading frozen fixture..."
    )

    index, chunks = load_index(
        FIXTURE
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    retriever = FaissRetriever(
        model=model,
        index=index,
        chunks=chunks,
    )

    llm = LocalLLM()

    print(
        "Warming up LLM..."
    )

    llm.generate(
        prompt=(
            "Réponds uniquement "
            "par OK."
        )
    )

    m4_results = []
    m5_results = []

    for item in questions:
        print(
            f"\n{item['id']} | "
            f"{item['category']}"
        )

        # M4
        m4 = evaluate_m4(
            item,
            retriever,
            llm,
        )

        m4_results.append(m4)

        print(
            "M4 | "
            f"behavior="
            f"{m4['response_behavior_match']} | "
            f"time="
            f"{m4['generation_time']:.2f}s"
        )

        # M5
        m5 = evaluate_m5(
            item,
            retriever,
            llm,
        )

        m5_results.append(m5)

        if (
            m5[
                "grounding_supported"
            ]
            is True
        ):
            grounding = "SUPPORTED"

        elif (
            m5[
                "grounding_supported"
            ]
            is False
        ):
            grounding = "UNSUPPORTED"

        else:
            grounding = "SKIPPED"

        print(
            "M5 | "
            f"behavior="
            f"{m5['response_behavior_match']} | "
            f"accepted="
            f"{m5['final_acceptance_match']} | "
            f"citations="
            f"{m5['has_citations']} | "
            f"citation_ids_valid="
            f"{m5['citations_in_range']} | "
            f"grounding="
            f"{grounding} | "
            f"time="
            f"{m5['total_time']:.2f}s"
        )

    # Aggregation
    m4_summary = aggregate_m4(
        m4_results
    )

    m5_summary = aggregate_m5(
        m5_results
    )

    comparison = build_comparison(
        m4_summary,
        m5_summary,
    )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    M4_OUTPUT.write_text(
        json.dumps(
            {
                "summary": m4_summary,
                "results": m4_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    M5_OUTPUT.write_text(
        json.dumps(
            {
                "summary": m5_summary,
                "results": m5_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    SUMMARY_OUTPUT.write_text(
        json.dumps(
            comparison,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # Console summary
    # --------------------------------------------------------------

    print(
        "\n========== "
        "FINAL M4 VS M5 "
        "=========="
    )

    print(
        f"Questions             : "
        f"{len(questions)}"
    )

    print(
        "\n---------- "
        "RETRIEVAL "
        "----------"
    )

    for key in m4_summary[
        "retrieval"
    ]:
        print(
            f"{key:<12} | "
            f"M4="
            f"{m4_summary['retrieval'][key]:.4f} | "
            f"M5="
            f"{m5_summary['retrieval'][key]:.4f} | "
            f"delta="
            f"{comparison['delta']['retrieval'][key]:+.4f}"
        )

    print(
        "\n---------- "
        "BEHAVIOR "
        "----------"
    )

    print(
        "M4 response behavior  : "
        f"{m4_summary['response_behavior_match_rate']:.4f}"
    )

    print(
        "M5 response behavior  : "
        f"{m5_summary['response_behavior_match_rate']:.4f}"
    )

    print(
        "M5 final acceptance   : "
        f"{m5_summary['final_acceptance_match_rate']:.4f}"
    )

    print(
        "M5 citation presence  : "
        f"{m5_summary['citation_presence_rate']:.4f}"
    )

    print(
        "M5 invalid citations  : "
        f"{m5_summary['invalid_citation_count']}"
    )

    print(
        "M5 citation validation: "
        f"{m5_summary['citation_validation_pass_rate']:.4f}"
    )

    print(
        "M5 grounding checked  : "
        f"{m5_summary['grounding_checked']}"
    )

    print(
        "M5 grounding supported: "
        f"{m5_summary['grounding_supported']}"
    )

    print(
        "M5 grounding rejected : "
        f"{m5_summary['grounding_rejected']}"
    )

    print(
        "\n---------- "
        "LATENCY "
        "----------"
    )

    print(
        "M4 generation avg     : "
        f"{m4_summary['average_generation_time']:.2f}s"
    )

    print(
        "M5 generation avg     : "
        f"{m5_summary['average_generation_time']:.2f}s"
    )

    print(
        "M5 grounding avg      : "
        f"{m5_summary['average_grounding_time']:.2f}s"
    )

    print(
        "M5 total avg          : "
        f"{m5_summary['average_total_time']:.2f}s"
    )

    print(
        "\n---------- "
        "DELTA "
        "----------"
    )

    print(
        "Response behavior     : "
        f"{comparison['delta']['response_behavior_match_rate']:+.4f}"
    )

    print(
        "End-to-end latency    : "
        f"{comparison['delta']['average_total_time']:+.2f}s"
    )

    print(
        f"\nM4 results : "
        f"{M4_OUTPUT}"
    )

    print(
        f"M5 results : "
        f"{M5_OUTPUT}"
    )

    print(
        f"Comparison : "
        f"{SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":
    main()