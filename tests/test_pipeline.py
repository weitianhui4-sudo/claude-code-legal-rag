"""Unit tests for the data processing pipeline."""

import pytest
from pipeline.cleaning import clean_document
from pipeline.chunking import chunk_document
from pipeline.ingestion import detect_document_type
from pipeline.models import DocumentType, RawDocument


SAMPLE_CONTRACT = """\
SOFTWARE LICENSE AGREEMENT

This Agreement is entered into as of January 15, 2024 by and between:
LICENSOR: TechCorp Solutions, Inc. ("Licensor")
LICENSEE: Global Enterprises LLC ("Licensee")

WHEREAS, Licensor has developed certain proprietary software; and

ARTICLE 1 - DEFINITIONS
1.1 "Software" means the computer programs identified in Exhibit A.
1.2 "Term" means three (3) years from the Effective Date.

ARTICLE 2 - LICENSE GRANT
2.1 Subject to the terms of this Agreement, Licensor grants Licensee a non-exclusive license.
2.2 Licensee shall not reverse engineer the Software.

IN WITNESS WHEREOF, the parties have executed this Agreement.
"""

SAMPLE_STATUTE = """\
CONSUMER DATA PRIVACY ACT
§ 45-12-1. Short Title
The Legislature finds that privacy is a fundamental right.
§ 45-12-2. Definitions
(1) "Consumer" means a natural person.
(2) "Personal data" means any information linkable to a person.
§ 45-12-3. Enforcement
The Attorney General shall enforce this Chapter.
"""

SAMPLE_CASE = """\
IN THE UNITED STATES DISTRICT COURT
SMITH v. APEX FINANCIAL SERVICES
Case No. 23-CV-04521
MEMORANDUM OPINION AND ORDER ON MOTION FOR SUMMARY JUDGMENT
JUDGE PATRICIA WASHINGTON

I. INTRODUCTION
Plaintiff brings this class action against Defendants.

II. LEGAL STANDARD
Summary judgment is appropriate under Fed. R. Civ. P. 56(a).

SO ORDERED.
"""


def make_raw(text: str, doc_type: DocumentType) -> RawDocument:
    return RawDocument(
        doc_id="test-id",
        filename="test.txt",
        raw_text=text,
        doc_type=doc_type,
        source_path="",
    )


class TestDetection:
    def test_contract_detected(self):
        assert detect_document_type(SAMPLE_CONTRACT) == DocumentType.CONTRACT

    def test_statute_detected(self):
        assert detect_document_type(SAMPLE_STATUTE) == DocumentType.STATUTE

    def test_case_law_detected(self):
        assert detect_document_type(SAMPLE_CASE) == DocumentType.CASE_LAW


class TestCleaning:
    def test_ocr_ligatures_fixed(self):
        raw = make_raw("The ﬁnal clause of the agreement.", DocumentType.CONTRACT)
        cleaned = clean_document(raw)
        assert "fi" in cleaned.cleaned_text
        assert "ﬁ" not in cleaned.cleaned_text

    def test_page_numbers_removed(self):
        text = "Some clause.\n\n42\n\nAnother clause."
        raw = make_raw(text, DocumentType.CONTRACT)
        cleaned = clean_document(raw)
        # Standalone "42" line should be gone
        lines = cleaned.cleaned_text.splitlines()
        assert "42" not in lines

    def test_citation_normalized(self):
        raw = make_raw("See 15 U.S.C.§78a.", DocumentType.STATUTE)
        cleaned = clean_document(raw)
        assert "15 U.S.C. § 78a" in cleaned.cleaned_text

    def test_metadata_contract(self):
        # Use the full raw doc which has "Effective Date" in proper format
        from pathlib import Path
        raw_text = Path("data/raw/contract_software_license.txt").read_text()
        raw = make_raw(raw_text, DocumentType.CONTRACT)
        cleaned = clean_document(raw)
        assert "effective_date" in cleaned.metadata

    def test_stats_present(self):
        raw = make_raw(SAMPLE_CONTRACT, DocumentType.CONTRACT)
        cleaned = clean_document(raw)
        assert "original_char_count" in cleaned.processing_stats
        assert "cleaned_char_count" in cleaned.processing_stats


class TestChunking:
    def test_contract_chunks_by_article(self):
        raw = make_raw(SAMPLE_CONTRACT, DocumentType.CONTRACT)
        from pipeline.cleaning import clean_document as cd
        cleaned = cd(raw)
        chunks = chunk_document(cleaned)
        assert len(chunks) >= 1
        titles = [c.section_title for c in chunks]
        # At least one chunk should have an ARTICLE heading
        assert any("ARTICLE" in t.upper() or "PREAMBLE" in t.upper() for t in titles)

    def test_statute_chunks_by_section(self):
        raw = make_raw(SAMPLE_STATUTE, DocumentType.STATUTE)
        from pipeline.cleaning import clean_document as cd
        cleaned = cd(raw)
        chunks = chunk_document(cleaned)
        assert len(chunks) >= 1

    def test_chunks_have_ids(self):
        raw = make_raw(SAMPLE_CONTRACT, DocumentType.CONTRACT)
        from pipeline.cleaning import clean_document as cd
        cleaned = cd(raw)
        chunks = chunk_document(cleaned)
        for c in chunks:
            assert c.chunk_id
            assert c.doc_id == "test-id"

    def test_no_empty_chunks(self):
        raw = make_raw(SAMPLE_CONTRACT, DocumentType.CONTRACT)
        from pipeline.cleaning import clean_document as cd
        cleaned = cd(raw)
        chunks = chunk_document(cleaned)
        for c in chunks:
            assert c.text.strip(), f"Empty chunk found: {c.chunk_id}"
