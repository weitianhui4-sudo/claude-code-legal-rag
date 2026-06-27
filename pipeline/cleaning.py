"""Text cleaning for legal documents.

Cleaning stages:
  1. Normalize whitespace and line endings
  2. Remove page headers/footers (page numbers, running heads)
  3. Fix OCR artifacts (common ligatures, broken hyphens)
  4. Normalize legal citation formats
  5. Preserve structural markers (section numbers, article headings)
"""

from __future__ import annotations

import re

from pipeline.models import CleanedDocument, DocumentType, RawDocument


# ── Stage helpers ──────────────────────────────────────────────────────────────

def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of blank lines to a maximum of two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces from each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def _remove_page_artifacts(text: str) -> str:
    # Standalone page numbers: lines containing only a number (optionally "Page N of M")
    text = re.sub(r"(?m)^\s*Page\s+\d+\s+of\s+\d+\s*$", "", text)
    text = re.sub(r"(?m)^\s*-\s*\d+\s*-\s*$", "", text)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    return text


def _fix_ocr_artifacts(text: str) -> str:
    # Common ligature replacements from PDF extraction
    replacements = {
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "–": "-", "—": "--", " ": " ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    # Fix soft hyphens at end of lines (word wrap artifacts)
    text = re.sub(r"-\n(\w)", r"\1", text)
    return text


def _normalize_citations(text: str) -> str:
    # Standardize U.S.C. references: "15 U.S.C.§78a" -> "15 U.S.C. § 78a"
    text = re.sub(r"(\d+)\s*U\.S\.C\.?\s*§\s*(\w+)", r"\1 U.S.C. § \2", text)
    # Add space after § if missing
    text = re.sub(r"§(\w)", r"§ \1", text)
    return text


def _extract_metadata(text: str, doc_type: DocumentType) -> dict:
    """Pull structured fields from the document header."""
    meta: dict = {}

    if doc_type == DocumentType.CONTRACT:
        m = re.search(r"as of\s+([A-Z][a-z]+ \d{1,2},\s*\d{4})", text, re.I) or \
            re.search(r"Effective Date[:\s]+([A-Z][a-z]+ \d{1,2},\s*\d{4})", text, re.I)
        if m:
            meta["effective_date"] = m.group(1).strip()
        parties = re.findall(r'"(\w[\w\s]+?)"[,\s]*\("(?:Licensor|Licensee|Party|Buyer|Seller)"\)', text)
        if parties:
            meta["parties"] = parties

    elif doc_type == DocumentType.CASE_LAW:
        m = re.search(r"Case No\.\s*([\w\-]+)", text, re.I)
        if m:
            meta["case_number"] = m.group(1).strip()
        m = re.search(r"JUDGE\s+([A-Z][A-Z\s]+)\n", text)
        if m:
            meta["judge"] = m.group(1).strip().title()

    elif doc_type == DocumentType.STATUTE:
        m = re.search(r'"([^"]+ACT[^"]*)"', text, re.I)
        if m:
            meta["act_name"] = m.group(1).strip()

    elif doc_type == DocumentType.MEMO:
        for field in ("TO", "FROM", "DATE", "RE"):
            m = re.search(rf"^{field}:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
            if m:
                meta[field.lower()] = m.group(1).strip()

    return meta


# ── Public API ─────────────────────────────────────────────────────────────────

def clean_document(raw: RawDocument) -> CleanedDocument:
    """Run all cleaning stages on a RawDocument and return a CleanedDocument."""
    text = raw.raw_text
    original_len = len(text)

    text = _fix_ocr_artifacts(text)
    text = _remove_page_artifacts(text)
    text = _normalize_citations(text)
    text = _normalize_whitespace(text)

    meta = _extract_metadata(text, raw.doc_type)
    meta.update({
        "filename": raw.filename,
        "doc_type": raw.doc_type.value,
        "source_path": raw.source_path,
    })

    stats = {
        "original_char_count": original_len,
        "cleaned_char_count": len(text),
        "reduction_pct": round((1 - len(text) / max(original_len, 1)) * 100, 1),
    }

    return CleanedDocument(
        doc_id=raw.doc_id,
        filename=raw.filename,
        cleaned_text=text,
        doc_type=raw.doc_type,
        metadata=meta,
        processing_stats=stats,
    )


def clean_documents(raws: list[RawDocument]) -> list[CleanedDocument]:
    return [clean_document(r) for r in raws]
