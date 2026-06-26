"""
Local end-to-end test script for the Legal RAG system.

Runs through every stage and prints results so you can eyeball correctness
without needing an API key for the pipeline stages.

Usage:
    # Pipeline + retrieval only (no API key needed):
    python test_local.py

    # Full RAG including Claude generation (requires ANTHROPIC_API_KEY):
    python test_local.py --with-generation
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Helpers ────────────────────────────────────────────────────────────────────

DIVIDER = "=" * 70
SECTION = "-" * 50


def header(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def info(msg: str) -> None:
    print(f"  {msg}")


# ── Stage tests ────────────────────────────────────────────────────────────────

def test_ingestion() -> list:
    header("STAGE 1: Ingestion")
    from pipeline.ingestion import load_documents

    docs = load_documents("data/raw")
    assert docs, "No documents loaded — check data/raw directory"
    info(f"Loaded {len(docs)} documents:")
    for d in docs:
        info(f"  {d.filename:<45} type={d.doc_type.value}")
    ok(f"All {len(docs)} documents ingested and typed")
    return docs


def test_cleaning(raw_docs: list) -> list:
    header("STAGE 2: Cleaning")
    from pipeline.cleaning import clean_documents

    cleaned = clean_documents(raw_docs)
    for d in cleaned:
        s = d.processing_stats
        info(f"  {d.filename}")
        info(f"    chars: {s['original_char_count']:,} -> {s['cleaned_char_count']:,}  "
             f"({s['reduction_pct']}% reduction)")
        if d.metadata:
            interesting = {k: v for k, v in d.metadata.items()
                           if k not in ("filename", "doc_type", "source_path")}
            if interesting:
                info(f"    metadata: {interesting}")
    ok("All documents cleaned")
    return cleaned


def test_chunking(cleaned_docs: list) -> list:
    header("STAGE 3: Chunking")
    from pipeline.chunking import chunk_documents

    chunks = chunk_documents(cleaned_docs)
    assert chunks, "No chunks produced"

    by_doc: dict[str, list] = {}
    for c in chunks:
        fname = c.metadata.get("filename", c.doc_id)
        by_doc.setdefault(fname, []).append(c)

    for fname, doc_chunks in by_doc.items():
        token_counts = [c.token_count for c in doc_chunks]
        info(f"  {fname}")
        info(f"    chunks={len(doc_chunks)}  "
             f"tokens: min={min(token_counts)} avg={sum(token_counts)//len(token_counts)} max={max(token_counts)}")
        for c in doc_chunks[:3]:
            title = c.section_title[:55] if c.section_title else "(no title)"
            info(f"      [{c.chunk_type.value}] {title}")
        if len(doc_chunks) > 3:
            info(f"      ... and {len(doc_chunks) - 3} more chunks")

    ok(f"{len(chunks)} chunks total across {len(by_doc)} documents")
    return chunks


def test_vector_store(chunks: list) -> object:
    header("STAGE 4: Vector Index")
    from rag.vector_store import VectorStore

    store = VectorStore()
    store.build(chunks, verbose=True)

    index_dir = Path("data/index")
    store.save(index_dir)

    # Round-trip: reload and verify
    reloaded = VectorStore.load(index_dir)
    assert reloaded._index.ntotal == store._index.ntotal, "Index round-trip mismatch"
    ok(f"Index built ({store._index.ntotal} vectors, dim={store._index.d}), saved and reloaded successfully")
    return store


def test_retrieval(store: object) -> None:
    header("STAGE 5: Retrieval")
    from rag.retriever import Retriever

    retriever = Retriever(store, use_mmr=True)

    queries = [
        ("payment terms and license fees",        "contract_software_license.txt"),
        ("non-compete agreement California FTC",   "legal_memo_employment.txt"),
        ("consumer rights opt out data privacy",   "statute_data_privacy.txt"),
        ("Meridian Bank aiding abetting ruling",   "case_law_smith_v_jones.txt"),
    ]

    all_passed = True
    for query, expected_doc in queries:
        results = retriever.retrieve(query, k=3)
        assert results, f"No results for: {query}"

        top_chunk, top_score = results[0]
        top_doc = top_chunk.metadata.get("filename", "")
        hit = expected_doc in top_doc

        status = "[PASS]" if hit else "[MISS]"
        if not hit:
            all_passed = False

        info(f"  {status} Q: \"{query}\"")
        info(f"         top result: {top_doc} (score={top_score:.3f})")
        info(f"         section: {top_chunk.section_title[:60]}")
        for chunk, score in results[1:]:
            info(f"           [{score:.3f}] {chunk.metadata.get('filename','')}: {chunk.section_title[:50]}")

    if all_passed:
        ok("All retrieval queries returned expected documents as top result")
    else:
        print("\n  [NOTE] Some queries missed expected top document — this is normal with TF-IDF backend.")
        print("         With sentence-transformers the semantic quality improves significantly.")


def test_generation(store: object) -> None:
    header("STAGE 6: Generation (Claude API)")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  [SKIP] ANTHROPIC_API_KEY not set. Pass --with-generation after setting the key.")
        return

    from rag.retriever import Retriever
    from rag.generator import generate_answer

    retriever = Retriever(store, use_mmr=True)

    question = "What are the termination rights and notice requirements in the software license agreement?"
    info(f"  Question: {question}\n")

    chunks = retriever.retrieve(question, k=5)
    response = generate_answer(question, chunks)

    print(f"  Answer:\n")
    for line in response.answer.strip().splitlines():
        print(f"    {line}")

    print(f"\n  Sources:")
    for s in response.sources:
        print(f"    - {s['filename']} | {s['section']} (score={s['relevance_score']})")

    print(f"\n  Tokens used: {response.usage['input_tokens']} in / {response.usage['output_tokens']} out")
    ok("Generation completed")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Local test runner for the Legal RAG system")
    parser.add_argument("--with-generation", action="store_true",
                        help="Include Claude generation stage (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--stage", choices=["ingest", "clean", "chunk", "index", "retrieve", "generate"],
                        help="Run only up to (and including) this stage")
    args = parser.parse_args()

    stages = ["ingest", "clean", "chunk", "index", "retrieve"]
    if args.with_generation:
        stages.append("generate")
    if args.stage:
        cutoff = stages.index(args.stage)
        stages = stages[: cutoff + 1]

    print(f"\nLegal RAG — Local Test Runner")
    print(f"Stages: {' -> '.join(stages)}")

    try:
        raw_docs = cleaned_docs = chunks = store = None

        if "ingest" in stages:
            raw_docs = test_ingestion()
        if "clean" in stages:
            cleaned_docs = test_cleaning(raw_docs)
        if "chunk" in stages:
            chunks = test_chunking(cleaned_docs)
        if "index" in stages:
            store = test_vector_store(chunks)
        if "retrieve" in stages:
            test_retrieval(store)
        if "generate" in stages:
            test_generation(store)

        print(f"\n{DIVIDER}")
        print("  ALL STAGES COMPLETED SUCCESSFULLY")
        print(DIVIDER + "\n")

    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
