from pathlib import Path
from time import perf_counter

from sentence_transformers import SentenceTransformer

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
    save_analysis,
    save_index,
    save_manifest,
    update_index,
)
from app.parser import chunking, read_dir
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
    save_analysis(memory_dir, question, answer)

    print(f"\n{answer}")
    stats = llm.last_stats

    if stats:
        title = f"Test {test_number} summary" if test_number is not None else "Question summary"
        print("\n=================================")
        print(title)
        print("=================================")
        print(f"Prompt tokens    : {stats['prompt_tokens']}")
        print(f"Generated tokens : {stats['completion_tokens']}")
        print(f"Total tokens     : {stats['total_tokens']}")
        print(f"Response time    : {elapsed:.2f} s")


def run_manual(pipeline, llm, memory_dir):
    print("\nManual mode")
    print("Press Enter without a question to quit.")

    while True:
        question = input("\nQuestion : ").strip()

        if not question:
            break

        ask_question(pipeline, llm, memory_dir, question)


def run_auto(pipeline, llm, memory_dir):
    print("\nAutomatic test")
    print(f"{len(AUTO_QUESTIONS)} questions will be tested.")
    print("Warming up model...")

    llm.generate("Réponds uniquement par OK.")

    for number, question in enumerate(AUTO_QUESTIONS, start=1):
        ask_question(pipeline, llm, memory_dir, question, test_number=number)


def main():
    model = SentenceTransformer(EMBEDDING_MODEL)
    memory_dir = project_memory_dir(PROJECT_PATH, MEMORY_ROOT)

    files = read_dir(PROJECT_PATH)
    current_manifest = build_manifest(files, PROJECT_PATH)

    if index_exists(memory_dir):
        print("Loading index from memory...")
        index, chunks = load_index(memory_dir)

        if manifest_exists(memory_dir):
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

                changed_paths = (
                    changes["added"]
                    + changes["modified"]
                    + changes["deleted"]
                )

                files_by_path = {
                    project_relative_path(file["path"], PROJECT_PATH): file
                    for file in files
                }

                new_chunks = []

                for path in changes["added"] + changes["modified"]:
                    file = files_by_path[path]
                    new_chunks.extend(chunking(file))

                index, chunks = update_index(
                    model=model,
                    index=index,
                    chunks=chunks,
                    new_chunks=new_chunks,
                    changed_paths=changed_paths,
                    project_path=PROJECT_PATH,
                )

                save_index(index, chunks, memory_dir)
                save_manifest(current_manifest, memory_dir)

                print(
                    f"\nIndex updated: "
                    f"{len(changes['added'])} added, "
                    f"{len(changes['modified'])} modified, "
                    f"{len(changes['deleted'])} deleted."
                )
            else:
                print("No project changes detected.")
        else:
            save_manifest(current_manifest, memory_dir)
            print("Project state initialized.")
    else:
        print("Building index...")
        chunks = []

        for file in files:
            chunks.extend(chunking(file))

        index = build_index(
            model=model,
            chunks=chunks,
            dimension=EMBEDDING_DIMENSION,
        )

        save_index(index, chunks, memory_dir)
        save_manifest(current_manifest, memory_dir)

    retriever = FaissRetriever(
        model=model,
        index=index,
        chunks=chunks,
    )

    llm = LocalLLM()

    pipeline = RAGPipeline(
        retriever=retriever.retrieve,
        llm=llm,
        top_k=TOP_K,
    )

    print("\nChoose mode:")
    print("[0] - Manual")
    print("[1] - Automatic test")

    mode = input("\nChoice : ").strip()

    if mode == "1":
        run_auto(pipeline, llm, memory_dir)
    else:
        run_manual(pipeline, llm, memory_dir)


if __name__ == "__main__":
    main()