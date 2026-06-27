"""Document ingestion: load PDF files from disk, detect document type."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import fitz  # PyMuPDF

from pipeline.models import DocumentType, RawDocument


_TYPE_PATTERNS: list[tuple[DocumentType, list[str]]] = [
    (DocumentType.CONTRACT, [
        r"\bAGREEMENT\b", r"\bLICENS(E|OR)\b", r"\bLICENSEE\b",
        r"\bIN WITNESS WHEREOF\b", r"\bEFFECTIVE DATE\b",
        r"\bRECITALS\b", r"\bWHEREAS\b",
    ]),
    (DocumentType.CASE_LAW, [
        r"\bPLAINTIFF\b", r"\bDEFENDANT\b", r"\bCOURT\b",
        r"\bMEMORANDUM OPINION\b", r"\bSUMMARY JUDGMENT\b",
        r"\bSO ORDERED\b", r"\bCase No\.",
    ]),
    (DocumentType.STATUTE, [
        r"\bSTATUTE\b", r"\bACT\b", r"\bLEGISLATUR\b",
        r"§\s*\d+", r"\bSECTION\s+\d+\b", r"\bCHAPTER\s+\d+\b",
        r"\bSHALL\b.*\bMEAN\b",
    ]),
    (DocumentType.MEMO, [
        r"\bMEMORANDUM\b", r"\bTO:\s+\w", r"\bFROM:\s+\w",
        r"\bRE:\s+\w", r"\bATTORNEY.CLIENT\b",
    ]),
]


def detect_document_type(text: str) -> DocumentType:
    upper = text[:3000].upper()
    scores: dict[DocumentType, int] = {dt: 0 for dt, _ in _TYPE_PATTERNS}
    for doc_type, patterns in _TYPE_PATTERNS:
        for pat in patterns:
            if re.search(pat, upper):
                scores[doc_type] += 1
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] >= 2 else DocumentType.UNKNOWN


def _extract_text_from_pdf(path: Path) -> str:
    pdf = fitz.open(str(path))
    text = "\n".join(page.get_text() for page in pdf)
    pdf.close()
    return text


def load_documents(data_dir: str | Path) -> list[RawDocument]:
    """Load all PDF files from data_dir into RawDocument objects."""
    data_dir = Path(data_dir)
    docs: list[RawDocument] = []

    pdf_files = sorted(data_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"[ingestion] No PDF files found in {data_dir}")
        return docs

    for path in pdf_files:
        text = _extract_text_from_pdf(path)
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(path)))
        doc_type = detect_document_type(text)
        docs.append(RawDocument(
            doc_id=doc_id,
            filename=path.name,
            raw_text=text,
            doc_type=doc_type,
            source_path=str(path),
        ))

    return docs
