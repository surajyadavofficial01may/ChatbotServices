# AI Knowledge Assistant — production image.
#   docker build -t ai-knowledge-assistant .
#   docker run --env-file .env -p 8000:8000 ai-knowledge-assistant
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/app/.fastembed_cache \
    APP_ENV=production \
    PORT=8000

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the default embedding models into the image so containers start without
# downloading them. A different EMBEDDING_MODEL/SPARSE_MODEL downloads at first use.
RUN python -c "from fastembed import SparseTextEmbedding, TextEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5'); SparseTextEmbedding('Qdrant/bm25')"

COPY app ./app

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Platforms (Render, Railway, Fly, a load balancer) terminate TLS at a proxy that sets
# X-Forwarded-For; trusting it keeps the per-IP rate limit per visitor. If you expose
# this container directly to the internet with no proxy, drop --forwarded-allow-ips.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --forwarded-allow-ips='*'"]
