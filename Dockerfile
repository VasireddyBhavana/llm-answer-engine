FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    GROUNDED_DB=/var/lib/grounded/grounded.db \
    GROUNDED_TRACE_FILE=/var/lib/grounded/traces.jsonl \
    GROUNDED_MEMORY_FILE=/var/lib/grounded/memory.json

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir . \
 && useradd --system --uid 10001 app \
 && mkdir -p /var/lib/grounded && chown app /var/lib/grounded

# Run as an unprivileged user; data lives in a volume so it survives container restarts.
USER app
VOLUME ["/var/lib/grounded"]
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "grounded.api:app", "--host", "0.0.0.0", "--port", "8000"]
