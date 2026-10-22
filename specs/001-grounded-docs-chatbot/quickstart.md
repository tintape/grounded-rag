# Quickstart validation

Prereqs: Docker (or Python 3.12 + uv), an OpenAI API key in `.env`.

1. `cp .env.example .env` and set `OPENAI_API_KEY`.
2. `docker compose up -d qdrant api ui`
3. `docker compose run --rm ingest` → report shows 4 loaded documents, 0 failed.
4. Open http://localhost:8501, ask "How do I factory reset the S7?" → answer cites
   `product-guide.pdf` with a page number.
5. Ask "What is the CEO's salary?" → refusal, no sources.
6. Re-run step 3 → `/health` chunk count unchanged.
7. `docker compose run --rm api grounded-rag eval` → `eval/results/report.md` with refusal accuracy
   and source hit rate.

Offline checks (no key): `pytest --cov` passes with coverage ≥ 80%.
