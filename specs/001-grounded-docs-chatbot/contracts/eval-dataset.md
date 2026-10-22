# Contract: Evaluation dataset (`eval/questions.json`)

```json
[
  {
    "id": "q01",
    "question": "How long is the warranty on the battery?",
    "reference_answer": "12 months from purchase.",
    "expected_sources": [{ "doc_name": "warranty-and-returns.pdf", "page": 2 }],
    "answerable": true
  },
  { "id": "q17", "question": "Who is the CEO?", "answerable": false }
]
```
Rules: `id` unique; `question` non-empty; `answerable=false` cases must not have
`expected_sources`. Report written to `eval/results/report.json` and `eval/results/report.md`.
