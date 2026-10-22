# Research: Grounded Document Q&A Chatbot

## R1. Vector store

- **Decision**: Qdrant.
- **Rationale**: runs as one Docker container next to the app (data stays with the client, no
  extra SaaS subscription); first-class payload filtering and deletion by `doc_id`; the Python
  client has an embedded local mode, so dev and tests need no server.
- **Alternatives**: pgvector (good if Postgres already exists — easy swap behind `store.py`);
  Chroma (fine for prototypes, weaker ops story); Pinecone (managed, but overkill and a recurring
  cost for ~100 PDFs).

## R2. Framework

- **Decision**: LlamaIndex for node parsing, vector-store integration, and OpenAI LLM/embedding
  clients; our own thin engine for grounding logic.
- **Rationale**: LlamaIndex's document/node model maps cleanly to page metadata. Writing the
  answer step ourselves (instead of a stock query engine) keeps the refusal gate and citation
  validation explicit and unit-testable.
- **Alternatives**: LangChain (equivalent capability, more indirection for this use case); stock
  `CitationQueryEngine` (citations are not validated and refusals are prompt-only).

## R3. PDF parsing and chunking

- **Decision**: PyMuPDF page-by-page text extraction with whitespace normalisation; chunks built
  per page with `SentenceSplitter` (default 512 tokens, 64 overlap); pages with < 20 chars of text
  are skipped; a file whose pages all lack text is reported as `skipped_no_text` (likely scanned).
- **Rationale**: per-page chunking makes page citations exact by construction. Sentence-aware
  splitting avoids cutting steps of an SOP mid-sentence.
- **Alternatives**: cross-page chunking (better context continuity, but ambiguous page
  citations); `unstructured`/Docling (better tables and OCR, much heavier — documented as an
  upgrade path for scanned PDFs).

## R4. Idempotent ingestion

- **Decision**: `doc_id = uuid5(namespace, file_name)`; chunk id = `uuid5(doc_id, f"{page}:{idx}")`.
  Before inserting a document, delete all points with that `doc_id`, then insert.
- **Rationale**: re-ingest of unchanged files is a no-op in effect (same ids, same count); edited
  files fully replace old passages, so stale pages cannot linger.

## R5. Grounding strategy (refusals)

- **Decision**: two layers.
  1. Retrieval gate: drop passages below `MIN_RELEVANCE` (cosine, default 0.20); if none remain,
     refuse without calling the LLM.
  2. Generation contract: prompt requires `[n]` citations for every claim and the exact token
     `NOT_FOUND` when context is insufficient. Citations outside `1..k` are discarded; no valid
     citation ⇒ refusal.
- **Rationale**: layer 1 removes the cheapest, most common hallucination path (off-topic
  questions); layer 2 covers "related but unanswered" questions. Validation ensures returned
  sources are always real retrieved passages.
- **Tuning**: `MIN_RELEVANCE` and `TOP_K` are env-configurable and re-checked with the eval set.
- **Measured (2026-10-23)**: unanswerable on-domain questions scored 0.36–0.59, above some
  answerable ones (0.31); only off-topic input scored low (0.05). Layer 2 does the precision work,
  so the default was lowered from 0.30 to 0.20 to protect recall.

## R6. Models

- **Decision**: `text-embedding-3-small` for embeddings; chat model configurable via
  `OPENAI_CHAT_MODEL` (default `gpt-4.1-mini`), temperature 0.
- **Rationale**: low cost, strong enough for extractive support answers; deterministic output
  aids evaluation. Model names change over time, hence configuration rather than code.

## R7. Evaluation

- **Decision**: built-in metrics always computed — refusal accuracy (answerable vs unanswerable),
  source hit rate (expected document + page appears in citations), p95 latency — plus RAGAS-style
  LLM-judged faithfulness and correctness implemented in-house (`evaluation/judge.py`).
- **Rationale**: `ragas` 0.4.3 fails on import against current `langchain-community`
  (`langchain_community.chat_models.vertexai` removed). Two judge prompts are ~80 lines and remove
  a fragile dependency chain from the quality gate.
- **Alternatives**: pin an older langchain stack for RAGAS (brittle); DeepEval (heavier).

## R8. Demo data

- **Decision**: generate synthetic PDFs for a fictional company ("Lumora Home Robotics", a robot
  vacuum maker): product guide, warranty & returns SOP, support FAQ, troubleshooting guide.
- **Rationale**: public demo without confidential or copyrighted material; known facts on known
  pages make the eval set precise.
