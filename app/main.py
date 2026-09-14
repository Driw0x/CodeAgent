from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.llm.local_llm import LocalLLM
from app.memory.vector_store import build_index
from app.parser.chunker import chunking
from app.parser.file_loader import read_dir
from app.rag.pipeline import RAGPipeline
from app.retrieval.faiss_retriever import FaissRetriever


PROJECT_PATH = Path(__file__).resolve().parents[1]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
TOP_K = 5


def main():
    model = SentenceTransformer(EMBEDDING_MODEL)

    files = read_dir(PROJECT_PATH)

    chunks = []

    for file in files:
        chunks.extend(chunking(file))

    index = build_index(
        model=model,
        chunks=chunks,
        dimension=EMBEDDING_DIMENSION,
    )

    retriever = FaissRetriever(
        model=model,
        index=index,
        chunks=chunks,
    )

    pipeline = RAGPipeline(
        retriever=retriever.retrieve,
        llm=LocalLLM(),
        top_k=TOP_K,
    )

    question = input("Question : ").strip()
    answer = pipeline.ask(question)

    print()
    print(answer)


if __name__ == "__main__":
    main()