# Legal RAG System

A production-quality Retrieval-Augmented Generation (RAG) pipeline for legal documents, built as a technical assessment demonstration for LexisNexis Global Content Management.

## Architecture

```
data/raw/            ← raw legal documents (.txt, .pdf)
    │
    ▼
pipeline/ingestion   ← load files, detect document type (contract/statute/case law/memo)
    │
    ▼
pipeline/cleaning    ← normalize whitespace, fix OCR artifacts, extract metadata
    │
    ▼
pipeline/chunking    ← section-aware splitting (articles, statutes, Roman-numeral sections)
    │                  + sliding-window fallback with overlap
    ▼
rag/embeddings       ← sentence-transformers (all-MiniLM-L6-v2, 384-dim, normalized)
    │
    ▼
rag/vector_store     ← FAISS IndexFlatIP (cosine similarity via dot product)
    │
    ▼
rag/retriever        ← hybrid: dense search + keyword boost + MMR diversification
    │
    ▼
rag/generator        ← Claude claude-sonnet-4-6 with grounded system prompt + source citation
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Section-aware chunking** | Legal documents have rigid hierarchical structure (ARTICLE 1 → 1.1 → 1.1(a)). Respecting these boundaries preserves the legal meaning that spans a clause. |
| **Sliding window fallback** | Long sections exceeding MAX_TOKENS=512 are split with OVERLAP_TOKENS=64 to avoid losing cross-sentence context at boundaries. |
| **Normalized embeddings + IndexFlatIP** | Normalizing vectors turns inner product into cosine similarity, giving consistent scores without IndexFlatL2. |
| **Keyword boost** | Legal queries often include specific statute numbers, case names, or dollar amounts. A small additive score for exact token matches improves precision without needing BM25. |
| **MMR diversification** | Without MMR, top-k results often cluster in one section. MMR ensures retrieved context spans different document parts. |
| **Grounded system prompt** | The generator is instructed to answer only from provided excerpts and cite sources in brackets, reducing hallucination on legal facts. |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

## Running

```bash
# Full pipeline: ingest -> clean -> chunk -> build index -> interactive query
python main.py

# Build index only
python main.py --build-only

# Query only (reuse existing index)
python main.py --query-only
```

## Testing

```bash
pytest tests/ -v
```

## Sample Documents

| File | Type | Description |
|---|---|---|
| `contract_software_license.txt` | Contract | Enterprise SaaS license agreement |
| `case_law_smith_v_jones.txt` | Case Law | Securities fraud class action ruling |
| `statute_data_privacy.txt` | Statute | Model state consumer data privacy act |
| `legal_memo_employment.txt` | Memo | Attorney memo on non-compete enforceability |

## Example Queries

- "What are the payment terms and late fees in the software license?"
- "What was the court's ruling on the aiding and abetting claim against Meridian Bank?"
- "What rights does a consumer have under the data privacy statute?"
- "Can we enforce non-compete agreements in California after the FTC rule?"
