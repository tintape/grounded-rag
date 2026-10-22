---
description: "Task list for Grounded Document Q&A Chatbot"
---

# Tasks: Grounded Document Q&A Chatbot

**Input**: Design documents from `/specs/001-grounded-docs-chatbot/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Required by constitution Principle III (test-first, offline fakes, ≥ 80% coverage).

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [X] T001 Initialise repo, spec-kit, `pyproject.toml` with runtime/ui/eval/dev extras
- [X] T002 [P] Configure ruff + pytest + coverage in `pyproject.toml`
- [X] T003 [P] Add `.gitignore`, `.env.example`, `.dockerignore`

## Phase 2: Foundational

- [X] T004 Settings with validation in `src/grounded_rag/config.py` (+ `tests/unit/test_config.py`)
- [X] T005 Frozen domain models in `src/grounded_rag/models.py` (+ `tests/unit/test_models.py`)
- [X] T006 Synthetic demo PDFs generator `scripts/make_sample_pdfs.py` → `data/sample_pdfs/`

## Phase 3: User Story 3 — Load and refresh the library (P2, prerequisite for US1/US2)

- [X] T007 [P] [US3] Tests for PDF reading (text pages, blank pages, scanned, corrupt) in `tests/unit/test_pdf_reader.py`
- [X] T008 [US3] Implement `src/grounded_rag/ingest/pdf_reader.py`
- [X] T009 [P] [US3] Tests for page-bounded chunking and deterministic ids in `tests/unit/test_chunker.py`
- [X] T010 [US3] Implement `src/grounded_rag/ingest/chunker.py`
- [X] T011 [US3] Tests for store replace/search/count on in-memory Qdrant in `tests/integration/test_store.py`
- [X] T012 [US3] Implement `src/grounded_rag/store.py`
- [X] T013 [US3] Tests for folder ingest report + idempotency in `tests/integration/test_pipeline.py`
- [X] T014 [US3] Implement `src/grounded_rag/ingest/pipeline.py`

## Phase 4: User Stories 1 & 2 — Cited answers and honest refusals (P1) 🎯 MVP

- [X] T015 [P] [US1] Tests for citation parsing/validation in `tests/unit/test_citations.py`
- [X] T016 [US1] Implement `src/grounded_rag/rag/citations.py`
- [X] T017 [US1] [US2] Tests for engine: threshold refusal, NOT_FOUND, invalid citations, cited-only sources, empty store, upstream error in `tests/unit/test_engine.py`
- [X] T018 [US1] [US2] Implement `src/grounded_rag/rag/prompts.py` and `src/grounded_rag/rag/engine.py`
- [X] T019 [US1] Wire real components in `src/grounded_rag/factory.py`

## Phase 5: User Story 5 — API and chat page (P3)

- [X] T020 [P] [US5] API tests (health, chat ok, refusal, 422, 503) in `tests/integration/test_api.py`
- [X] T021 [US5] Implement `src/grounded_rag/api/app.py`
- [X] T022 [US5] Implement `src/grounded_rag/ui/streamlit_app.py`

## Phase 6: User Story 4 — Evaluation (P2)

- [X] T023 [P] [US4] Tests for dataset validation, metrics and report in `tests/unit/test_evaluation.py`
- [X] T024 [US4] Implement `src/grounded_rag/evaluation/{dataset,runner,report}.py` (+ optional RAGAS)
- [X] T025 [US4] Demo question set `eval/questions.json` (answerable + unanswerable)

## Phase 7: Polish

- [X] T026 CLI `src/grounded_rag/cli.py` (ingest | ask | eval | serve) + tests
- [X] T027 [P] `Dockerfile`, `docker-compose.yml` (qdrant, api, ui, ingest)
- [X] T028 [P] GitHub Actions CI `.github/workflows/ci.yml` (ruff + pytest + coverage gate)
- [X] T029 README with architecture diagram, quickstart, design decisions, eval summary
- [X] T030 Code review pass; coverage ≥ 80%; ruff clean

## Dependencies

Setup → Foundational → US3 (ingest) → US1/US2 (engine) → US5 (API/UI) and US4 (eval) in
parallel → Polish.
