# Data Model

All models are immutable (frozen dataclasses or frozen pydantic models).

## PageText
| Field | Type | Rule |
|---|---|---|
| doc_name | str | PDF file name, e.g. `product-guide.pdf` |
| page | int | 1-based |
| text | str | whitespace-normalised, non-empty |

## Chunk (Passage)
| Field | Type | Rule |
|---|---|---|
| chunk_id | str | `uuid5(doc_id, f"{page}:{index}")` — deterministic |
| doc_id | str | `uuid5(NAMESPACE, doc_name)` |
| doc_name | str | |
| page | int | ≥ 1; a chunk never spans pages |
| index | int | position within the page, 0-based |
| text | str | non-empty |

Stored in Qdrant as point id = `chunk_id`, payload = `doc_id`, `doc_name`, `page`, `index`, text.

## RetrievedChunk
`chunk: Chunk`, `score: float` (cosine similarity, higher is better).

## Source (returned citation)
| Field | Type | Rule |
|---|---|---|
| doc_name | str | |
| page | int | |
| excerpt | str | ≤ 300 chars of the chunk text |
| score | float | retrieval score, rounded to 3 dp |

## Answer
| Field | Type | Rule |
|---|---|---|
| answer | str | answer text, or the refusal message |
| refused | bool | true ⇒ `sources` is empty |
| sources | tuple[Source] | only passages cited in `answer`, de-duplicated by (doc, page) |

Invariant: `refused == (len(sources) == 0)`.

## FileIngestResult / IngestReport
`FileIngestResult`: `doc_name`, `status ∈ {loaded, skipped_no_text, failed}`, `pages`, `chunks`,
`error: str | None`. `IngestReport`: tuple of results + totals.

## EvalCase / EvalResult
`EvalCase`: `id`, `question`, `reference_answer?`, `expected_sources?: [{doc_name, page}]`,
`answerable: bool` (default true).
`EvalResult`: case + produced `Answer` + `refusal_correct: bool` + `source_hit: bool | None` +
optional metric scores.
