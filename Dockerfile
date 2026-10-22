FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[ui]" \
    # warm the tokenizer cache so containers start without downloading it
    && python -c "from llama_index.core.node_parser import SentenceSplitter; SentenceSplitter().split_text('warm up')"

COPY data ./data
COPY eval ./eval

RUN useradd --create-home --uid 1000 app && chown -R app /app /opt/tiktoken
USER app

EXPOSE 8000 8501
CMD ["grounded-rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
