# Grounded RAG Constitution

## Core Principles

### I. Grounded or Silent (NON-NEGOTIABLE)

Every answer MUST be derived only from the ingested documents. When retrieval finds no
sufficiently relevant passage, the system MUST refuse with an explicit "not in the documents"
response and MUST NOT call the language model to improvise. An answer whose citations cannot be
mapped back to retrieved passages MUST be treated as a refusal.

Rationale: a support bot that guesses is worse than no bot; trust is the product.

### II. Every Claim Is Traceable

Each answer MUST carry citations naming the source document and page number. Citations MUST be
produced from stored chunk metadata, never generated free-form by the model. Only sources the
answer actually cites are returned to the caller.

Rationale: support staff must be able to verify an answer in seconds.

### III. Test-First With Fakes at the Boundaries

Core logic (PDF parsing, chunking, retrieval gating, citation mapping, refusal) MUST be covered by
unit tests written before implementation. External services (LLM, embeddings, vector store) MUST
sit behind small interfaces so tests run offline, fast, and without API keys. Line coverage of the
`grounded_rag` package MUST stay at or above 80%.

Rationale: RAG regressions are silent; only tests and evals make them visible.

### IV. Measured, Not Felt

Answer quality MUST be evaluated against a versioned question set that includes unanswerable
questions. Any change to chunking, prompts, thresholds, or models MUST be re-evaluated and the
summary committed alongside the change.

Rationale: tuning without measurement trades one failure mode for another.

### V. Simple, Swappable, Small

Prefer the simplest design that meets the spec. Each module stays focused (<400 lines), functions
small (<50 lines), data structures immutable (frozen dataclasses / pydantic models). Configuration
comes from environment variables with validated defaults; no secrets in code.

Rationale: the client must be able to own and extend the code after handover.

## Operational Constraints

- Python 3.12, FastAPI backend, Streamlit UI, Qdrant vector store, LlamaIndex, OpenAI API.
- The whole stack MUST start with a single `docker compose up` plus a `.env` file.
- Ingestion MUST be idempotent: re-ingesting the same PDFs replaces, never duplicates, chunks.
- Inputs at the API boundary MUST be validated (non-empty question, bounded length).
- Errors from upstream services MUST surface as clear HTTP errors, never as fabricated answers.

## Development Workflow

- Spec → plan → tasks → implement, tracked under `specs/`.
- Every task lands with its tests; `pytest` and `ruff` MUST pass before commit.
- Conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).

## Governance

This constitution supersedes ad-hoc preferences. Amendments require updating this file with a
version bump (semantic versioning: MAJOR for principle removal or redefinition, MINOR for new
principles, PATCH for wording) and re-checking open plans against the change. Reviews MUST verify
compliance with Principles I–IV before merge.

**Version**: 1.0.0 | **Ratified**: 2026-10-23 | **Last Amended**: 2026-10-23
