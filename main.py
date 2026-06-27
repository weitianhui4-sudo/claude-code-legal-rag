"""
Start the Legal RAG system locally and enter an interactive Q&A session.

Usage:
    python run.py
"""

import os
import sys
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

RAW_DIR = Path("data/raw")
INDEX_DIR = Path("data/index")


def build_index_if_needed() -> None:
    if not (INDEX_DIR / "embeddings.npy").exists():
        print("Index not found — building from raw documents...")
        from pipeline.pipeline import run_pipeline
        from rag.vector_store import VectorStore

        chunks = run_pipeline(RAW_DIR, Path("data/processed"), verbose=True)
        store = VectorStore()
        store.build(chunks)
        store.save(INDEX_DIR)
        print("Index ready.\n")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    build_index_if_needed()

    from rag.rag_pipeline import LegalRAG
    rag = LegalRAG()
    rag.load_index(INDEX_DIR)

    print("\n" + "=" * 60)
    print("  Legal RAG — Ask questions about your legal documents")
    print("  Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break

        try:
            response = rag.query(question, k=5)
        except Exception as e:
            print(f"Error: {e}\n")
            continue

        print(f"\nAnswer:\n{response.answer}")
        print("\nSources:")
        for s in response.sources:
            print(f"  - {s['filename']} | {s['section']} (score={s['relevance_score']})")
        print()


if __name__ == "__main__":
    main()
