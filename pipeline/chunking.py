"""Legal-aware document chunking.

Strategy:
  - For statutes and contracts: split on numbered sections/articles (structural chunking).
  - For case law: split on Roman-numeral sections (I. II. III. ...).
  - For memos: split on lettered sub-sections and paragraph breaks.
  - Fall back to sliding-window paragraph chunking when structure is absent.

Each chunk is limited to MAX_TOKENS tokens to fit embedding model windows.
Overlapping context (OVERLAP_TOKENS) is prepended from the previous chunk
to preserve cross-boundary coherence.
"""

from __future__ import annotations

import re
import uuid
from typing import Iterator

from pipeline.models import ChunkType, CleanedDocument, DocumentChunk, DocumentType


MAX_TOKENS = 512
OVERLAP_TOKENS = 64

# A crude but dependency-free token estimator (≈ GPT tokenization: 1 token ≈ 4 chars)
def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ── Section splitters ──────────────────────────────────────────────────────────

_ARTICLE_RE = re.compile(
    r"(?m)^(ARTICLE\s+\d+|SECTION\s+\d+|\d+\.\d*\s+[A-Z]|§\s*[\d\-]+\.\s*\w)",
    re.IGNORECASE,
)

_ROMAN_RE = re.compile(
    r"(?m)^(I{1,4}V?|VI{0,4}|IX|X{1,3})\.\s+[A-Z\s]{3,}$"
)

_MEMO_SECTION_RE = re.compile(
    r"(?m)^([IVX]+\.|[A-Z]\.|[A-Z]{2,}\.)\s+[A-Z]"
)


def _split_by_pattern(text: str, pattern: re.Pattern) -> list[tuple[str, str]]:
    """Return list of (heading, body) pairs by splitting on pattern matches."""
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    sections = []
    for i, m in enumerate(matches):
        heading = m.group(0).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((heading, body))
    # Prepend any preamble before first match
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.insert(0, ("PREAMBLE", preamble))
    return sections


def _sliding_window_chunks(text: str, max_tokens: int, overlap_tokens: int) -> Iterator[str]:
    """Split text into overlapping chunks by paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    current: list[str] = []
    current_tokens = 0
    overlap_buffer: list[str] = []

    for para in paragraphs:
        para_tokens = _approx_tokens(para)
        if current_tokens + para_tokens > max_tokens and current:
            yield "\n\n".join(current)
            # Keep last few paragraphs as overlap
            overlap_buffer = []
            overlap_total = 0
            for p in reversed(current):
                t = _approx_tokens(p)
                if overlap_total + t > overlap_tokens:
                    break
                overlap_buffer.insert(0, p)
                overlap_total += t
            current = overlap_buffer[:]
            current_tokens = sum(_approx_tokens(p) for p in current)
        current.append(para)
        current_tokens += para_tokens

    if current:
        yield "\n\n".join(current)


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_document(doc: CleanedDocument) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    text = doc.cleaned_text
    doc_type = doc.doc_type

    # Choose section pattern based on document type
    if doc_type in (DocumentType.CONTRACT,):
        sections = _split_by_pattern(text, _ARTICLE_RE)
        chunk_type = ChunkType.ARTICLE
    elif doc_type == DocumentType.STATUTE:
        sections = _split_by_pattern(text, _ARTICLE_RE)
        chunk_type = ChunkType.SECTION
    elif doc_type == DocumentType.CASE_LAW:
        sections = _split_by_pattern(text, _ROMAN_RE)
        chunk_type = ChunkType.SECTION
    elif doc_type == DocumentType.MEMO:
        sections = _split_by_pattern(text, _MEMO_SECTION_RE)
        chunk_type = ChunkType.SECTION
    else:
        sections = []
        chunk_type = ChunkType.PARAGRAPH

    if sections:
        for idx, (heading, body) in enumerate(sections):
            # If section body is too long, further split with sliding window
            if _approx_tokens(body) > MAX_TOKENS:
                for sub_idx, sub_text in enumerate(_sliding_window_chunks(body, MAX_TOKENS, OVERLAP_TOKENS)):
                    chunk_id = str(uuid.uuid4())
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        text=sub_text,
                        chunk_type=chunk_type,
                        section_title=heading,
                        chunk_index=len(chunks),
                        token_count=_approx_tokens(sub_text),
                        metadata={**doc.metadata, "sub_chunk": sub_idx},
                    ))
            else:
                chunk_id = str(uuid.uuid4())
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    text=body,
                    chunk_type=chunk_type,
                    section_title=heading,
                    chunk_index=len(chunks),
                    token_count=_approx_tokens(body),
                    metadata=doc.metadata,
                ))
    else:
        # Fall back to sliding-window chunking
        for sub_text in _sliding_window_chunks(text, MAX_TOKENS, OVERLAP_TOKENS):
            chunk_id = str(uuid.uuid4())
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=sub_text,
                chunk_type=ChunkType.PARAGRAPH,
                section_title="",
                chunk_index=len(chunks),
                token_count=_approx_tokens(sub_text),
                metadata=doc.metadata,
            ))

    return chunks


def chunk_documents(docs: list[CleanedDocument]) -> list[DocumentChunk]:
    all_chunks: list[DocumentChunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
