"""
Legal RAG System - Main Entry Point

Usage:
    # Build the pipeline + index, then run an interactive query session:
    python main.py

    # Build only:
    python main.py --build-only

    # Query only (requires existing index):
    python main.py --query-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("data/index")


def build(verbose: bool = True) -> None:
    from pipeline.pipeline import run_pipeline
    from rag.rag_pipeline import LegalRAG

    chunks = run_pipeline(RAW_DIR, PROCESSED_DIR, verbose=verbose)

    print("\n[Main] Building vector index...")
    rag = LegalRAG()
    rag.build_index(chunks, INDEX_DIR)
    print("[Main] Index ready.\n")


def interactive_query() -> None:
    from rag.rag_pipeline import LegalRAG

    rag = LegalRAG()
    rag.load_index(INDEX_DIR)

    print("\n" + "=" * 60)
    print("  Legal RAG System — Interactive Query Mode")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            break

        print("\nSearching...\n")
        try:
            response = rag.query(question, k=5)
        except Exception as exc:
            print(f"[Error] {exc}\n")
            continue

        print("=" * 60)
        print(response.answer)
        print("\n--- Sources ---")
        for s in response.sources:
            print(f"  [{s['filename']} - {s['section']}] (score={s['relevance_score']})")
        print(f"\n[Tokens used: {response.usage['input_tokens']} in / {response.usage['output_tokens']} out]")
        print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal RAG System")
    parser.add_argument("--build-only", action="store_true", help="Build index and exit")
    parser.add_argument("--query-only", action="store_true", help="Skip build, load existing index")
    parser.add_argument("--quiet", action="store_true", help="Suppress pipeline verbosity")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY") and not args.build_only:
        print("[Warning] ANTHROPIC_API_KEY not set. Query mode will fail.")

    if not args.query_only:
        build(verbose=not args.quiet)

    if not args.build_only:
        interactive_query()


if __name__ == "__main__":
    main()
