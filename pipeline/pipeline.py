"""Orchestrates the full data processing pipeline: ingest -> clean -> chunk."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.chunking import chunk_documents
from pipeline.cleaning import clean_documents
from pipeline.ingestion import load_documents
from pipeline.models import DocumentChunk


def run_pipeline(
    raw_dir: str | Path,
    output_dir: str | Path,
    verbose: bool = True,
) -> list[DocumentChunk]:
    """
    Full pipeline: load raw docs -> clean -> chunk -> save to output_dir.
    Returns the list of DocumentChunk objects ready for embedding.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Ingestion
    if verbose:
        print(f"\n[Pipeline] Stage 1: Ingesting documents from {raw_dir}")
    raw_docs = load_documents(raw_dir)
    if verbose:
        print(f"  Loaded {len(raw_docs)} documents")
        for d in raw_docs:
            print(f"    {d.filename} -> type={d.doc_type.value}")

    # Stage 2: Cleaning
    if verbose:
        print("\n[Pipeline] Stage 2: Cleaning documents")
    cleaned_docs = clean_documents(raw_docs)
    if verbose:
        for d in cleaned_docs:
            s = d.processing_stats
            print(f"  {d.filename}: {s['original_char_count']} -> {s['cleaned_char_count']} chars "
                  f"({s['reduction_pct']}% reduction)")

    # Stage 3: Chunking
    if verbose:
        print("\n[Pipeline] Stage 3: Chunking documents")
    chunks = chunk_documents(cleaned_docs)
    if verbose:
        print(f"  Produced {len(chunks)} chunks total")
        by_doc: dict[str, int] = {}
        for c in chunks:
            by_doc[c.metadata.get("filename", c.doc_id)] = by_doc.get(
                c.metadata.get("filename", c.doc_id), 0
            ) + 1
        for fname, count in by_doc.items():
            print(f"    {fname}: {count} chunks")

    # Persist chunks as JSON for downstream use
    chunks_path = output_dir / "chunks.json"
    chunks_data = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "text": c.text,
            "chunk_type": c.chunk_type.value,
            "section_title": c.section_title,
            "chunk_index": c.chunk_index,
            "token_count": c.token_count,
            "metadata": c.metadata,
        }
        for c in chunks
    ]
    chunks_path.write_text(json.dumps(chunks_data, indent=2, ensure_ascii=False))
    if verbose:
        print(f"\n[Pipeline] Chunks saved to {chunks_path}")

    return chunks
