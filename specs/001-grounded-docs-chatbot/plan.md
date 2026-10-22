# Implementation Plan: Grounded Document Q&A Chatbot

**Branch**: `001-grounded-docs-chatbot` | **Date**: 2026-10-23 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-grounded-docs-chatbot/spec.md`

## Summary

Page-aware RAG over a folder of PDFs. PDFs are parsed page by page (PyMuPDF), split into
overlapping passages that never cross a page boundary (LlamaIndex `SentenceSplitter`), embedded
with OpenAI embeddings and stored in Qdrant with document/page metadata and deterministic IDs
(idempotent re-ingest). At question time a **two-layer grounding gate** applies: (1) a retrieval
relevance threshold refuses before any LLM call; (2) the LLM must cite numbered passages or emit a
`NOT_FOUND` sentinel, and citations are validated against the retrieved set. Only validated,
cited passages are returned as sources. Served via FastAPI `/chat`, consumed by a Streamlit page,
measured by an eval runner (refusal accuracy + source hit rate, optional RAGAS faithfulness and
answer relevancy). Shipped with Docker Compose and synthetic demo PDFs.

## Technical Context

**Language/Version**: Python 3.12

**Primary Dependencies**: LlamaIndex (core, Qdrant store, OpenAI LLM/embeddings), qdrant-client,
PyMuPDF, FastAPI + Uvicorn, pydantic-settings, Streamlit, httpx; optional RAGAS for eval

**Storage**: Qdrant (Docker service in prod; embedded local mode for dev/tests)

**Testing**: pytest + pytest-cov; LlamaIndex `MockEmbedding` and in-memory Qdrant for store tests;
hand-written fakes for retriever/LLM in engine and API tests

**Target Platform**: Linux containers (Docker Compose); macOS/Linux for local dev

**Project Type**: web service + CLI + thin web UI

**Performance Goals**: answer p95 < 10 s (dominated by one LLM call); ingest 80 PDFs < 5 min

**Constraints**: offline test suite (no API keys); answers only from retrieved context; no
secrets in repo

**Scale/Scope**: ~100 PDFs, ~5–20k passages, single tenant, internal users

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Gate | Status |
|---|---|---|
| I. Grounded or Silent | Retrieval threshold gate before LLM; `NOT_FOUND` sentinel; invalid-citation → refusal | PASS |
| II. Every Claim Traceable | Sources built from stored metadata, filtered to cited indices only | PASS |
| III. Test-First w/ Fakes | `Retriever` / `AnswerGenerator` protocols; offline tests; coverage ≥ 80% gate in CI | PASS |
| IV. Measured, Not Felt | `eval/questions.json` with unanswerable cases; committed report | PASS |
| V. Simple, Swappable, Small | One package, frozen models, env config, modules < 400 lines | PASS |

Post-design re-check (after Phase 1): all gates still PASS; no complexity violations.

## Project Structure

### Documentation (this feature)

```text
specs/001-grounded-docs-chatbot/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── chat-api.md
│   └── eval-dataset.md
└── tasks.md
```

### Source Code (repository root)

```text
src/grounded_rag/
├── config.py              # Settings from env (validated)
├── models.py              # Frozen domain models: PageText, Chunk, Source, Answer, reports
├── ingest/
│   ├── pdf_reader.py      # PDF → per-page cleaned text + file status
│   ├── chunker.py         # pages → page-bounded chunks with deterministic ids
│   └── pipeline.py        # folder → store, per-file report
├── store.py               # Qdrant-backed vector store via LlamaIndex (upsert/replace/search)
├── rag/
│   ├── prompts.py         # system + user prompt templates
│   ├── citations.py       # parse/validate [n] markers
│   └── engine.py          # GroundedAnswerer: gate → generate → validate → Answer
├── factory.py             # wires real OpenAI/Qdrant components from Settings
├── api/app.py             # FastAPI app: /health, /chat
├── ui/streamlit_app.py    # chat page calling the API
├── evaluation/
│   ├── dataset.py         # load/validate question set
│   ├── runner.py          # run cases, compute metrics
│   └── report.py          # JSON + Markdown report
└── cli.py                 # grounded-rag ingest | ask | eval | serve

scripts/make_sample_pdfs.py  # builds synthetic demo PDFs (fictional company)
data/sample_pdfs/            # generated demo PDFs (committed)
eval/questions.json          # demo question set incl. unanswerable cases
tests/unit/ , tests/integration/
Dockerfile, docker-compose.yml, .env.example, .github/workflows/ci.yml
```

**Structure Decision**: single Python package under `src/` with a CLI entry point; API and UI are
thin layers over `GroundedAnswerer`, so all business logic is testable without HTTP or network.

## Complexity Tracking

No constitution violations to justify.
