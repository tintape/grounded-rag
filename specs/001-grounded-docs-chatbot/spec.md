# Feature Specification: Grounded Document Q&A Chatbot

**Feature Branch**: `001-grounded-docs-chatbot`

**Created**: 2026-10-23

**Status**: Draft

**Input**: User description: "A support chatbot that answers questions only from a company's own
PDF documents (product guides, SOPs, FAQs), cites document name and page for every answer, says
'not in our documents' instead of guessing, exposes a chat API and a simple chat page, ships with
an answer-quality evaluation on a sample question set, and runs from a single container setup."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ask a question, get a cited answer (Priority: P1)

A support agent types a question in plain English ("How do I reset the device to factory
settings?"). The chatbot answers in a few sentences using only the company's documents and lists
the exact documents and pages the answer came from.

**Why this priority**: This is the core value: agents stop searching PDFs by hand and can verify
each answer instantly.

**Independent Test**: Load the sample documents, ask a question whose answer is on a known page,
and confirm the answer is correct and cites that document and page.

**Acceptance Scenarios**:

1. **Given** documents are loaded, **When** an agent asks a question answered on page 4 of the
   product guide, **Then** the answer reflects that content and cites "product guide, page 4".
2. **Given** an answer draws on two documents, **When** it is shown, **Then** both documents and
   their pages are cited, and no uncited sources are listed.

---

### User Story 2 - Refuse when the documents don't know (Priority: P1)

An agent asks something the documents do not cover ("What is your CEO's salary?"). The chatbot
clearly says the answer is not in the documents instead of inventing one.

**Why this priority**: A confident wrong answer to a customer is worse than no answer; this is
the main reason the company wants a grounded bot instead of a generic one.

**Independent Test**: Ask questions that are deliberately outside the documents and confirm every
one gets the "not in the documents" response with no citations.

**Acceptance Scenarios**:

1. **Given** documents are loaded, **When** an agent asks an off-topic question, **Then** the
   chatbot responds that the information is not in the documents and lists no sources.
2. **Given** a question is loosely related but not actually answered by the documents, **When**
   it is asked, **Then** the chatbot refuses rather than giving a partial guess.

---

### User Story 3 - Load and refresh the document library (Priority: P2)

An operator drops PDFs into a folder and runs one command. The library is (re)built; running it
again after adding or editing a PDF updates the library without duplicating content.

**Why this priority**: Without a reliable load step there is nothing to answer from, but it is an
operator task performed rarely.

**Independent Test**: Run the load command twice on the same folder and confirm the library size
is unchanged; add one PDF and confirm only its content is added.

**Acceptance Scenarios**:

1. **Given** a folder of PDFs, **When** the operator runs the load command, **Then** a summary
   reports documents, pages, and passages loaded, and any files that could not be read.
2. **Given** the library is already loaded, **When** the command is run again, **Then** no
   duplicate passages are created.

---

### User Story 4 - Measure answer quality (Priority: P2)

The project owner provides 15–20 sample questions with expected answers (including some that are
deliberately unanswerable). One command runs them all and produces a short readable report on
correctness, grounding, and refusal behaviour.

**Why this priority**: The client must be able to see quality, not take it on faith, and re-check
it after any change.

**Independent Test**: Run the evaluation on the sample set and confirm a report is produced with
per-question results and overall scores.

**Acceptance Scenarios**:

1. **Given** a question set file, **When** the evaluation runs, **Then** a report lists each
   question, the answer, its citations, and pass/fail on refusal expectations, plus summary scores.

---

### User Story 5 - Integrate via API and use a simple chat page (Priority: P3)

Developers call a chat endpoint from other tools; agents use a minimal web chat page that shows
answers with clickable-looking source chips.

**Why this priority**: Packaging for daily use and future integrations; core value is already
delivered by stories 1–2.

**Independent Test**: Send a request to the chat endpoint and receive an answer with structured
sources; open the chat page and ask a question.

**Acceptance Scenarios**:

1. **Given** the service is running, **When** a client posts a question, **Then** it receives the
   answer, a refusal flag, and a structured list of sources (document, page, excerpt).
2. **Given** an empty or overly long question, **When** it is posted, **Then** the service rejects
   it with a clear validation message.

### Edge Cases

- A PDF is scanned (image-only) and has no extractable text: it is reported as skipped with a
  reason during loading, not silently ignored.
- A PDF is corrupted or password-protected: reported as failed; other files still load.
- A page is blank or contains only headers/footers: no empty passages are stored.
- The answer generator cites a source number that was not provided: that citation is dropped; if
  no valid citation remains, the response becomes a refusal.
- The upstream AI service is unavailable or rate-limited: the user gets a clear temporary-error
  message, never a fabricated answer.
- The library is empty (nothing loaded yet): questions get a clear "no documents loaded" refusal.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load all PDF files from a configured folder, extracting text per page
  and recording document name and page number for every passage.
- **FR-002**: System MUST split page text into overlapping passages sized for retrieval while
  never merging text from different pages into one passage.
- **FR-003**: Loading MUST be idempotent: re-loading an unchanged document creates no duplicates;
  re-loading a changed document replaces its previous passages.
- **FR-004**: System MUST report per-file outcomes on load (loaded / skipped-no-text / failed).
- **FR-005**: For each question, system MUST retrieve the most relevant passages and MUST refuse
  without generating an answer when none meets a configurable relevance threshold.
- **FR-006**: Generated answers MUST use only retrieved passages and MUST cite them by reference.
- **FR-007**: System MUST return only sources actually cited in the answer, each with document
  name, page number, and a short excerpt.
- **FR-008**: If the generator indicates the passages don't contain the answer, or produces no
  valid citation, system MUST return the standard refusal message with no sources.
- **FR-009**: System MUST expose a chat endpoint accepting a question (1–2,000 characters) and
  returning answer text, a refusal flag, and sources; plus a health endpoint.
- **FR-010**: System MUST provide a chat web page that displays answers and their sources.
- **FR-011**: System MUST provide an evaluation command that runs a question set (with optional
  expected answers and an "unanswerable" marker) and writes a JSON and Markdown report.
- **FR-012**: Evaluation MUST report refusal accuracy (answerable answered, unanswerable refused)
  and, when an evaluation library is available, faithfulness and answer-relevancy scores.
- **FR-013**: The full system MUST start from one container-composition command with
  configuration supplied via an environment file.
- **FR-014**: Upstream service failures MUST produce explicit error responses, never answers.

### Key Entities

- **Document**: a source PDF; attributes: file name, content fingerprint, page count.
- **Passage (Chunk)**: a contiguous span of one page's text; attributes: text, document name,
  page number, passage index, stable identifier derived from document and position.
- **Source Citation**: a reference returned with an answer: document name, page, excerpt,
  relevance score.
- **Answer**: answer text, refusal flag, list of source citations.
- **Evaluation Case**: question, optional reference answer, expected-refusal flag; result adds
  produced answer, citations, pass/fail, and metric scores.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On the sample question set, at least 90% of answerable questions are answered with
  at least one citation pointing to the correct document and page.
- **SC-002**: 100% of deliberately unanswerable questions in the sample set receive the refusal
  response with no sources.
- **SC-003**: An agent receives an answer in under 10 seconds for 95% of questions.
- **SC-004**: A new operator can go from clean machine to a working chat page by following the
  README in under 15 minutes.
- **SC-005**: Re-running the load step on unchanged documents changes the library size by 0.

## Assumptions

- Documents are primarily text-based PDFs in English; scanned PDFs are detected and reported, and
  OCR is out of scope for v1.
- Single tenant, internal use: authentication, per-user history, and rate limiting are out of
  scope for v1 and are expected to be provided by the hosting environment if needed.
- The client supplies an API key for the hosted AI service and accepts its per-use cost.
- The demo ships with a small set of synthetic documents for a fictional company, so the repo is
  runnable and evaluable without any confidential data.
- Conversation is single-turn (each question answered independently); multi-turn memory is out
  of scope for v1.
