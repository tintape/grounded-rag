# Contract: Chat API

## GET /health
200 → `{"status": "ok", "chunks": <int>}` (number of passages in the store)

## POST /chat
Request:
```json
{ "question": "How do I reset the vacuum to factory settings?" }
```
- `question`: string, trimmed length 1–2000, required.

200 Response:
```json
{
  "answer": "Hold the Home and Spot buttons for 10 seconds until the ring flashes blue [1].",
  "refused": false,
  "sources": [
    { "doc_name": "product-guide.pdf", "page": 4, "excerpt": "Factory reset: hold ...", "score": 0.712 }
  ]
}
```
Refusal (still 200):
```json
{ "answer": "I couldn't find this in our documents.", "refused": true, "sources": [] }
```

Errors:
- 422 — validation failure (empty or too long question).
- 503 — upstream AI service or vector store unavailable: `{"detail": "..."}`. Never a fabricated answer.
