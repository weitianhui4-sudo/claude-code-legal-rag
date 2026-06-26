"""Shared data models for the legal document pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class DocumentType(str, Enum):
    CONTRACT = "contract"
    CASE_LAW = "case_law"
    STATUTE = "statute"
    MEMO = "memo"
    REGULATION = "regulation"
    UNKNOWN = "unknown"


class ChunkType(str, Enum):
    SECTION = "section"
    PARAGRAPH = "paragraph"
    ARTICLE = "article"
    EXHIBIT = "exhibit"


@dataclass
class RawDocument:
    doc_id: str
    filename: str
    raw_text: str
    doc_type: DocumentType = DocumentType.UNKNOWN
    source_path: str = ""
    ingested_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CleanedDocument:
    doc_id: str
    filename: str
    cleaned_text: str
    doc_type: DocumentType
    metadata: dict = field(default_factory=dict)
    processing_stats: dict = field(default_factory=dict)


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_type: ChunkType
    section_title: str = ""
    chunk_index: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)
