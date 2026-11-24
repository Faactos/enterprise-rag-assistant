# Enterprise RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant for answering questions over enterprise documents. Documents are chunked, embedded and stored in a Qdrant vector store; relevant chunks are retrieved (optionally with hybrid search and re-ranking) and passed to an LLM to generate grounded answers.

> **Status:** the repository is being set up — source code will be added soon.

## Features

- **Pluggable LLM providers** – [Groq](https://console.groq.com) (free hosted API) or fully offline via [Ollama](https://ollama.com)
- **Vector search with Qdrant** – embedded local store by default (no Docker), or point to a Qdrant server
- **Hybrid retrieval** – combine dense vector search with keyword search
- **Optional re-ranking** – re-score candidate chunks before passing them to the LLM
- **Tunable retrieval** – chunk size, overlap, top-k and score thresholds are configurable and can be swept to compare retrieval quality

## Getting started

### 1. Clone

```bash
git clone https://github.com/Faactos/enterprise-rag-assistant.git
cd enterprise-rag-assistant
```

### 2. Configure environment

Create a `.env` file in the project root. **Never commit it** — it is already listed in `.gitignore`.

```env
# LLM (Groq) – get a free key from https://console.groq.com/keys
GROQ_API_KEY=your-groq-api-key
LLM_PROVIDER=groq
LLM_MODEL=your-model-name

# Fully offline alternative (no key, needs `ollama pull llama3.1:8b`)
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.1:8b

# Retrieval settings
CHUNK_SIZE=
CHUNK_OVERLAP=
TOP_K=
CANDIDATE_K=
USE_HYBRID=
USE_RERANKER=
MIN_SCORE=

# Leave unset to use the embedded local Qdrant store
# QDRANT_URL=http://localhost:6333
```

### Configuration reference

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key for Groq (only needed when `LLM_PROVIDER=groq`) |
| `LLM_PROVIDER` | `groq` or `ollama` |
| `LLM_MODEL` | Model name used with Groq |
| `OLLAMA_MODEL` | Local model name used with Ollama |
| `CHUNK_SIZE` | Size of each document chunk |
| `CHUNK_OVERLAP` | Overlap between consecutive chunks |
| `TOP_K` | Number of chunks passed to the LLM |
| `CANDIDATE_K` | Number of candidates retrieved before re-ranking |
| `USE_HYBRID` | Enable hybrid (dense + keyword) retrieval |
| `USE_RERANKER` | Enable re-ranking of candidates |
| `MIN_SCORE` | Minimum relevance score for a chunk to be used |
| `QDRANT_URL` | Qdrant server URL; leave unset for the embedded store |

## Security

- Secrets live only in `.env`, which is git-ignored.
- Do not commit API keys, tokens or credentials. If a key is ever exposed, rotate it immediately.
