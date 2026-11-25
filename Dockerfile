FROM python:3.12-slim

# Build tools are needed for a few wheels; they are removed in the same layer
# so they do not ship in the image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding and reranking models into the image. Downloading ~1GB on
# first request is what made the original cold start time a client out.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

COPY app ./app
COPY ui ./ui
COPY eval ./eval
COPY packages ./packages
COPY data/corpus ./data/corpus

ENV PYTHONPATH=/srv:/srv/packages \
    PYTHONUNBUFFERED=1

EXPOSE 8000 8501
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
