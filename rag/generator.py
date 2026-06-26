"""Answer generation using Claude via the Anthropic API.

The generator:
  1. Formats retrieved chunks into a grounded context block.
  2. Builds a system prompt that instructs the model to act as a legal research
     assistant and cite sources.
  3. Calls Claude and returns a structured response with answer + citations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic

from pipeline.models import DocumentChunk


_SYSTEM_PROMPT = """\
You are a legal research assistant with expertise in contract law, case law, statutes, and regulatory compliance. \
You have been provided with relevant excerpts from legal documents.

Rules:
- Answer ONLY based on the provided document excerpts. Do not use outside knowledge.
- Cite your sources by referencing the document name and section title in square brackets, e.g. [contract_software_license.txt - ARTICLE 2].
- If the answer cannot be found in the provided excerpts, say so clearly.
- Be precise and use legal terminology appropriately.
- When quoting directly, use quotation marks and cite the source.
"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    query: str
    model: str
    usage: dict


def _format_context(chunks: list[tuple[DocumentChunk, float]]) -> str:
    parts = []
    for chunk, score in chunks:
        fname = chunk.metadata.get("filename", chunk.doc_id)
        section = chunk.section_title or "General"
        parts.append(
            f"--- SOURCE: {fname} | SECTION: {section} | relevance={score:.3f} ---\n"
            f"{chunk.text}\n"
        )
    return "\n".join(parts)


def generate_answer(
    query: str,
    retrieved_chunks: list[tuple[DocumentChunk, float]],
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 1024,
) -> RAGResponse:
    """Call Claude to answer query given retrieved chunks."""
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    context = _format_context(retrieved_chunks)
    user_message = f"Document excerpts:\n\n{context}\n\nQuestion: {query}"

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = response.content[0].text

    sources = [
        {
            "chunk_id": chunk.chunk_id,
            "filename": chunk.metadata.get("filename", ""),
            "section": chunk.section_title,
            "doc_type": chunk.metadata.get("doc_type", ""),
            "relevance_score": round(score, 4),
        }
        for chunk, score in retrieved_chunks
    ]

    return RAGResponse(
        answer=answer_text,
        sources=sources,
        query=query,
        model=response.model,
        usage={
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    )
