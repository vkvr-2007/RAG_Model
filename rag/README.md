# RAG Service Handoff and Integration Guide

This folder contains the working RAG service for the local Hindi MSMARCO-XI dataset. It is designed to be a small, practical API that a teammate can call from a backend or frontend after transcription. This service is intentionally conservative: it prefers a safe refusal (`grounded=false`) instead of pretending to know an answer when evidence is weak or missing.

This guide is written for a teammate who has never seen the code before. It explains exactly how to install, run, verify, and integrate the service without changing the existing architecture.

Important: the default endpoint is the fast retrieval path. The optional Groq generation path exists, but it is slower and is not used by default.

## 1) Project overview

### What this RAG service does

This service receives a transcribed user question, searches the local MSMARCO-XI-derived document chunks, and returns a grounded answer if the retrieved evidence supports it. If the evidence is weak, mismatched, or irrelevant, it refuses with a safe answer instead of hallucinating.

The core behavior is:

- accept a text query
- preprocess the query
- embed it
- retrieve candidates using FAISS
- retrieve candidate text using BM25
- combine the results with reciprocal-rank fusion (RRF)
- check whether the top evidence actually matches the user question
- return a grounded answer or a refusal

### What dataset it uses

This service uses the local Hindi MSMARCO-XI data already present in the repository under `rag/data/`.

Key runtime files:

- `rag/data/hindi_subset.parquet` — local subset used for indexing and retrieval
- `rag/data/hintrain.parquet` — larger local Hindi shard already present in the workspace
- `rag/data/index/` — built FAISS/BM25 metadata and vector index
- `rag/data/benchmark_queries_100.txt` — benchmark query set

This is a real local dataset, not synthetic mock data.

### Default request pipeline

Default path (fast retrieval path):

Voice
↓
STT handled by main application
↓
transcribed text
↓
POST /rag/query
↓
query validation
↓
query preprocessing
↓
embedding
↓
FAISS + BM25 retrieval
↓
RRF fusion
↓
relevance/grounding guard
↓
grounded answer or refusal
↓
main application/frontend

This is the default and it does not call an external LLM.

### Optional Groq generation path

Optional path when configured and explicitly triggered:

query
↓
retrieval
↓
context
↓
Groq OpenAI-compatible API
↓
generated answer
↓
grounding validation
↓
response

This path is slower than the default retrieval path and is only used when `?generate=true` is sent and the environment is configured.

### What is not included

This repository does not include:

- speech-to-text (STT)
- audio handling
- frontend UI code
- browser-only logic
- a full deployment stack

This service is intentionally just the RAG API layer. The teammate's application handles voice transcription and frontend display.

## 2) What the teammate receives in rag/

The `rag/` directory contains the working RAG service. It is intentionally isolated from the teammate's existing frontend/backend code.

Contents:

- `app/` — FastAPI app, retriever, indexer, generation, config, schemas, service logic
- `data/` — local dataset and runtime index files
- `tests/` — project tests for API, indexing, chunking, and retrieval
- `scripts/` — benchmark and local inspection utilities
- `requirements.txt` — runtime and test dependencies
- `.env.example` — template for local environment config
- `.gitignore` — excludes secrets and caches
- `README.md` — this guide
- No Dockerfile is currently present in this `rag/` directory

### Important files and folders

#### `app/`
This folder contains the actual service.

- `app/main.py` — FastAPI application, routes, CORS, health endpoint, validation
- `app/config.py` — environment configuration and settings
- `app/service.py` — retrieval logic, relevance guard, grounded answer logic, default fast path
- `app/retrieval.py` — FAISS/BM25/RRF retrieval and timing instrumentation
- `app/embeddings.py` — embedding backend loading and fallback logic
- `app/indexer.py` — local MSMARCO-XI indexing pipeline
- `app/chunking.py` — sentence/passage/recursive chunking strategies
- `app/generation.py` — optional OpenAI-compatible Groq generation path
- `app/schemas.py` — request/response models

#### `data/`
This folder stores both the local dataset and the index used at runtime.

- `data/hindi_subset.parquet` — local retrieval corpus for the current service
- `data/index/` — generated runtime artifacts:
  - `vectors.faiss`
  - `bm25.pkl`
  - `metadata.json`
- `data/benchmark_queries_100.txt` — measured local benchmark query list

#### `tests/`
The repository includes real tests for the API and retrieval flow.

- `tests/test_api.py`
- `tests/test_chunking.py`
- `tests/test_indexing_retrieval.py`
- `tests/test_service.py`

#### `scripts/`
Operational scripts to inspect data and run benchmark checks.

- `scripts/benchmark.py` — runs live HTTP latency benchmark against `/rag/query`
- helper scripts inspect the parquet structure and local dataset

#### `requirements.txt`
Lists the runtime dependencies used by the service.

#### `.env.example`
Template for local environment configuration. Copy it to `.env` and fill values.

#### `.gitignore`
Excludes `.env`, caches, temp files, and local Python artifacts.

## 3) System requirements

This project has been validated in the current environment using a modern Python interpreter. Use a current Python version; Python 3.11+ is the safe target.

Minimal practical requirements:

- Python: 3.11+ recommended
- OS: Windows, Linux, or macOS
- RAM: not strictly hard-coded; this local subset and index are small enough for regular local development
- Disk: a few hundred MB is enough for this subset and index; the full source parquet shard is much larger and is not required for the default runtime path
- Docker: optional at this point; there is no Dockerfile in `rag/` right now

The runtime dependencies are defined in `requirements.txt` and are installed via pip.

## 4) Fresh installation

Follow these steps on a fresh machine.

### Windows PowerShell

```powershell
cd rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux/macOS

```bash
cd rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

### Verify the Python environment

```bash
python --version
```

### Create the local environment file

You must create `.env` from `.env.example` before running. The service reads `.env` automatically via `pydantic-settings`.

## 5) Environment variables

The service reads variables from `.env` through `app/config.py`.

The environment variable names are:

- `RAG_INDEX_DIR`
- `RAG_EMBEDDING_BACKEND`
- `RAG_EMBEDDING_MODEL`
- `RAG_USE_BM25`
- `RAG_TOP_K`
- `RAG_MIN_RETRIEVAL_SCORE`
- `RAG_STRICT_EXTRACTIVE`
- `CORS_ALLOWED_ORIGINS`
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`
- `LLM_TIMEOUT_SECONDS`

### Required for the default RAG path

| Variable | Required? | Default | Purpose |
| --- | --- | --- | --- |
| `RAG_INDEX_DIR` | No | `data/index` | Runtime folder containing the built FAISS/BM25 index |
| `RAG_EMBEDDING_BACKEND` | No | `hash` | Embedding backend used at runtime |
| `RAG_EMBEDDING_MODEL` | No | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Model name used by sentence-transformers if selected |
| `RAG_USE_BM25` | No | `true` | Enables BM25 retrieval |
| `RAG_TOP_K` | No | `4` | Number of top candidates to return |
| `RAG_MIN_RETRIEVAL_SCORE` | No | `0.05` | Minimal retrieval threshold |
| `RAG_STRICT_EXTRACTIVE` | No | `true` | Keeps the service on the retrieval-only default path |
| `CORS_ALLOWED_ORIGINS` | No for local-only; yes for browser access from another origin | configured localhost defaults | Comma-separated allowed frontend origins |

### Optional for Groq generation

| Variable | Required? | Default | Purpose |
| --- | --- | --- | --- |
| `LLM_BASE_URL` | Only if using Groq generation | `None` | OpenAI-compatible base URL, e.g. `https://api.groq.com/openai/v1` |
| `LLM_API_KEY` | Only if using Groq generation | `None` | API key for external generation. It is required only when `?generate=true` is used or when the optional LLM generation feature is enabled. |
| `LLM_MODEL` | Only if using Groq generation | `None` | Model name, e.g. `openai/gpt-oss-20b` |
| `LLM_TIMEOUT_SECONDS` | Only if using Groq generation | `12` | Request timeout |

### Important notes

- `.env` is not committed to Git.
- Do not paste a real API key into README files or commit `.env`.
- The default request path does not require Groq. Groq is optional and slower.
- If you are not using the optional generation path, `LLM_API_KEY` does not need to be set.

## 6) Data and index

### Where the data is

The runtime corpus is stored under `rag/data/`.

- `rag/data/hindi_subset.parquet` is the working local subset used by the service
- `rag/data/hintrain.parquet` is the larger local Hindi shard already present in the workspace
- `rag/data/benchmark_queries_100.txt` is used for benchmark validation

### Where the index is

The runtime index files are stored in:

- `rag/data/index/`

Required runtime files:

- `vectors.faiss`
- `metadata.json`
- `bm25.pkl` if BM25 is enabled

### Whether the repository already contains a built index

Yes. The repository already includes a working index under `rag/data/index/`.

### Does the teammate need to rebuild it?

Usually not for a working local clone, because the repository already contains the runtime data and index files. If the index is deleted or the embedding config changes, then it must be rebuilt with the same backend/model configuration used by the application.

### Important embedding mismatch rule

The code uses `app/embeddings.py` to create the embedder. The project supports:

- `hash` backend
- `sentence_transformers` backend

`app/indexer.py` writes the index using the selected backend and model. `app/main.py` loads the same index and embeds query text at runtime. If the index was built with one backend/model and the service is started with a different one, retrieval may fail or behave incorrectly.

This is the correct rule:

- if the index was built with `hash`, keep runtime `RAG_EMBEDDING_BACKEND=hash`
- if the index was built with `sentence_transformers`, keep runtime `RAG_EMBEDDING_BACKEND=sentence_transformers` and the same model name

The code does have a fallback in `build_embedder()` that tries to fall back from `sentence_transformers` to `hash` if the sentence-transformers model cannot be loaded. That is useful for resilience, but it is not a safe way to silently mix different index and query embeddings. For predictable behavior, keep the backend/model pair consistent.

## 7) Start the API

### Local development

From the `rag/` folder:

```powershell
cd rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Linux/macOS:

```bash
cd rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Production / container / external host

Use `0.0.0.0` for the bind address in production. Do not bind to `127.0.0.1` when the service is expected to be reachable externally.

Example:

```bash
cd rag
source .venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

If the hosting platform sets the port externally, map that platform port to the application port you choose, usually 8000.

## 8) Health check

### GET /health

This endpoint checks whether the service is live and whether the local index was loaded.

### Curl example

```bash
curl http://127.0.0.1:8000/health
```

### PowerShell example

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -Method Get
```

### Expected result

```json
{
  "status": "ok",
  "index_loaded": true,
  "embedding_backend": "hash",
  "retrieval_ready": true,
  "detail": null
}
```

Field meanings:

- `status` — service status (`ok` or `degraded`)
- `index_loaded` — whether the app loaded the runtime index
- `embedding_backend` — selected embedding backend
- `retrieval_ready` — whether the retriever is initialized
- `detail` — startup message or error, if any

If the index is missing or startup fails, `status` may become `degraded` and `detail` will include the actual error.

## 9) RAG API contract

### Endpoint

`POST /rag/query`

### Request headers

```http
Content-Type: application/json
```

### Request body

```json
{
  "query": "भारत की राजधानी क्या है?"
}
```

### Response schema

The service response model is defined in `app/schemas.py`.

```json
{
  "answer": "string",
  "grounded": true,
  "confidence": 0.0,
  "sources": [],
  "latency_ms": 0
}
```

Field meanings:

- `answer` — the answer text or refusal text
- `grounded` — whether the response is grounded in retrieved evidence
- `confidence` — retrieval confidence value, expressed as a float between `0.0` and `1.0`
- `sources` — list of retrieved source chunks with metadata
- `latency_ms` — request latency in milliseconds

### Example successful response

```json
{
  "answer": "नई दिल्ली भारत की राजधानी है।",
  "grounded": true,
  "confidence": 0.9,
  "sources": [
    {
      "query_id": "123",
      "chunk_id": "1-0-0",
      "text": "नई दिल्ली भारत की राजधानी है।",
      "language": "hi",
      "chunking_strategy": "sentence",
      "score": 0.9
    }
  ],
  "latency_ms": 7.3
}
```

### Example grounded=false response

```json
{
  "answer": "संदर्भ में पर्याप्त जानकारी नहीं है",
  "grounded": false,
  "confidence": 0.0,
  "sources": [],
  "latency_ms": 8.1
}
```

### HTTP status codes

- `200 OK` — valid request and a normal response
- `400` — route-level empty/blank query guard
- `422` — malformed JSON, invalid payload, or validation failure
- `503` — index unavailable or startup initialization failed

### Empty and malformed requests

The API rejects:

- empty JSON body
- missing `query`
- empty or whitespace-only query
- non-string `query`
- unexpected extra fields (`extra="forbid"` in `QueryRequest`)

## 10) Frontend and backend integration

This service is designed to be called as a plain HTTP API. The teammate can either call it from the browser or from a backend service.

### Recommended approach

For production, call the RAG service from the backend instead of directly from a browser. This keeps secrets, URL configuration, and CORS policy out of the frontend.

If the frontend is running on a different origin (for example, localhost or a deployed web app), CORS must be configured on the RAG service.

### JavaScript example for browser integration

```javascript
async function askRAG(transcribedText) {
  const url = "http://localhost:8000/rag/query";

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ query: transcribedText })
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `RAG request failed: ${response.status}`);
    }

    const data = await response.json();

    if (!data.grounded) {
      return {
        answer: data.answer,
        grounded: false,
        confidence: data.confidence,
        sources: data.sources,
        latency_ms: data.latency_ms
      };
    }

    return data;
  } catch (error) {
    console.error("RAG request error:", error);
    throw error;
  }
}
```

### Example usage

```javascript
const transcribedText = "भारत की राजधानी क्या है?";
const result = await askRAG(transcribedText);
console.log(result.answer);
```

### Backend example

This is the recommended production pattern for many teams.

#### Node.js backend example

```javascript
const fetch = (...args) => import("node-fetch").then(({ default: fetch }) => fetch(...args));

async function askRAG(transcribedText) {
  const response = await fetch("http://localhost:8000/rag/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: transcribedText })
  });

  if (!response.ok) {
    throw new Error(`RAG HTTP error: ${response.status}`);
  }

  return response.json();
}
```

#### Python backend example

```python
import httpx

async def ask_rag(text: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/rag/query",
            json={"query": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
```

## 11) Voice and STT integration

This service does not accept audio directly. The teammate's frontend or main application is expected to do the STT step first.

The intended flow is:

audio
↓
Sarvam / ElevenLabs / another STT service
↓
transcribed text
↓
POST /rag/query
↓
answer

### Example

```javascript
const transcribedText = "भारत की राजधानी क्या है?";
const result = await askRAG(transcribedText);
console.log(result.answer);
```

This keeps the service focused on text-based retrieval and grounded answer generation.

## 12) CORS

The backend uses FastAPI CORS middleware configured in `app/main.py`.

The actual CORS settings are:

- allowed methods: `GET`, `POST`, `OPTIONS`
- allowed headers: `Content-Type`, `Authorization`
- credentials are disabled (`allow_credentials=False`)

The configuration comes from the environment variable `CORS_ALLOWED_ORIGINS`.

### Local development config

Example:

```env
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080
```

### Production config

For a deployed frontend, configure the exact deployed origin, for example:

```env
CORS_ALLOWED_ORIGINS=https://app.example.com
```

Do not use a wildcard origin or insecure credential configuration. The implementation is intentionally explicit and conservative.

## 13) Optional Groq generation

By default, the application does not call the external LLM. The default route is the fast retrieval path.

### Default behavior

```http
POST /rag/query
```

This calls the retrieval stack and returns the grounded answer from the local index. It does not call Groq.

### Optional slower generation mode

```http
POST /rag/query?generate=true
```

This sends the retrieved context to the optional OpenAI-compatible Groq endpoint if the environment is configured.

### Required environment variables for optional generation

```env
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your-api-key
LLM_MODEL=openai/gpt-oss-20b
LLM_TIMEOUT_SECONDS=12
```

### Important notes

- `LLM_API_KEY` is only required when the Groq generation path is being used.
- The Groq path is slower than the default retrieval endpoint.
- The default endpoint is the correct choice for low-latency integration.
- The project does not claim that the external LLM path is under 200ms.

## 14) Chunking

The indexer supports three chunking strategies, all implemented in `app/chunking.py`.

### 1. `passage`

Keeps each selected passage as a single chunk when possible.

### 2. `sentence`

Splits the passage on sentence boundaries and groups them into chunk windows with overlap.

### 3. `recursive`

Prefers paragraph and sentence boundaries while chunking into fixed-size windows.

### Metadata captured per chunk

Each chunk stores metadata including:

- `query_id`
- `chunk_id`
- `text`
- `source_passage`
- `language`
- `chunking_strategy`

These metadata values are used for retrieval and source reporting.

## 15) Retrieval architecture

The retrieval logic is in `app/retrieval.py` and `app/service.py`.

### Embedding

The query is embedded using the configured embedding backend. The service supports:

- `hash` fallback backend
- `sentence_transformers` backend

### FAISS

The API loads a FAISS index from `data/index/vectors.faiss`.

### BM25

The service loads BM25 from `data/index/bm25.pkl` when enabled.

### RRF fusion

The code combines dense and sparse retrieval results with reciprocal-rank fusion rather than comparing score scales directly. This is a standard retrieval fusion approach for mixed similarity methods.

### Top-k

The default config uses `RAG_TOP_K=4`.

### Relevance and grounding checks

The service does not trust a non-zero similarity score alone. It checks whether:

- the retrieved chunk contains the subject/entity terms from the query
- the top result is relevant enough to answer the question
- capital questions include both the capital relationship and the subject
- low-confidence or irrelevant results are refused

This is the guardrail that prevents false grounding.

## 16) Guardrails

The service is intentionally conservative.

Guardrails include:

- empty query rejection
- invalid payload rejection
- low-confidence rejection
- off-topic or unsupported query rejection
- subject mismatch rejection
- correct refusal for non-answering but semantically related passages
- generator fallback to a refusal when the optional LLM path fails

### Behavior on unsupported content

When the service cannot find enough relevant evidence, it returns:

```json
{
  "answer": "संदर्भ में पर्याप्त जानकारी नहीं है",
  "grounded": false,
  "confidence": 0.0,
  "sources": [],
  "latency_ms": 0
}
```

The system prefers refusal over hallucination.

## 17) Performance

These benchmark values are from the live validation of the default fast retrieval endpoint.

They do not include the optional Groq generation path.

| Metric | Value |
| --- | ---: |
| P50 | 5.68 ms |
| P70 | 5.88 ms |
| P95 | 6.46 ms |
| P100 | 23.92 ms |

This means the default endpoint is comfortably under the 200ms target. The optional Groq generation path is much slower and should not be treated as the default runtime path.

## 18) Testing

### Run the project test suite

```bash
cd rag
python -m pytest tests -q
```

### Run the benchmark

PowerShell:

```powershell
cd rag
python .\scripts\benchmark.py --queries-file .\data\benchmark_queries_100.txt
```

Linux/macOS:

```bash
cd rag
python scripts/benchmark.py --queries-file data/benchmark_queries_100.txt
```

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Manual query test

```bash
curl -X POST http://127.0.0.1:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"भारत की राजधानी क्या है?"}'
```

### Expected behavior for common validation cases

1. Valid answerable query
   - should return `grounded=true`
   - should return a relevant answer from the local retrieved context

2. Alabama regression
   - should not return an Arkansas passage as grounded
   - should return `grounded=false` when evidence is not relevant

3. Unrelated question
   - should return a refusal with `grounded=false`

The project has already validated these behaviors.

## 19) Deployment

There is no Dockerfile in the current `rag/` directory. The service is currently intended to run directly with `uvicorn`.

### Minimal deployment model

The simplest realistic deployment is:

- host the service on a Linux box or VM
- bind to `0.0.0.0:8000`
- keep `rag/data/` and `rag/data/index/` present
- configure `.env` for the chosen runtime settings
- set CORS for the actual frontend origin
- expose the backend URL to the main application

### Required deployment items

- Python environment
- `requirements.txt` installed
- `.env` present with required values
- `rag/data/index/` present
- `rag/data/hindi_subset.parquet` present if runtime needs the corpus
- `CORS_ALLOWED_ORIGINS` set for the deployed frontend domain
- secrets never committed to Git

### Production notes

- use `0.0.0.0`, not `127.0.0.1`, for external exposure
- make sure the platform can reach the service port
- do not expect the service to handle audio or browser-only features

## 20) GitHub and security

### Commit these files

These are expected to be tracked in Git:

- `rag/app/**`
- `rag/data/**` (when needed for the runtime service)
- `rag/tests/**`
- `rag/scripts/**`
- `rag/requirements.txt`
- `rag/.env.example`
- `rag/.gitignore`
- `rag/README.md`

### Never commit these

- `.env`
- API keys or credentials
- secret values
- virtual environments
- `__pycache__`
- cache directories
- temp output

### Create `.env` from `.env.example`

```bash
cp .env.example .env
```

PowerShell:

```powershell
Copy-Item .env.example .env
```

Then fill in only the values you actually need.

## 21) Troubleshooting

| Problem | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError` | dependencies not installed | run `pip install -r requirements.txt` |
| port already in use | another process is bound to 8000 | choose a different port or stop the conflicting process |
| health endpoint fails | startup index not loaded or wrong path | check `data/index` exists and the app can read the files |
| index missing | the generated index folder is absent or deleted | restore `data/index` or rebuild with the same config |
| embedding dimension mismatch | index and runtime embedding config do not match | keep the backend/model pair consistent |
| FAISS loading error | corrupt or missing `vectors.faiss` | restore the index or rebuild it |
| BM25 loading error | missing or incompatible `bm25.pkl` | verify `RAG_USE_BM25` and restore the index |
| CORS error | frontend origin not in `CORS_ALLOWED_ORIGINS` | add the exact frontend origin to the env var |
| frontend cannot connect | wrong RAG host or port | use the actual deployed URL or local host + port |
| `127.0.0.1` confusion | service is bound to localhost only | use `0.0.0.0` for external access |
| deployed service cannot find data files | working directory or mounted volume mismatch | ensure `rag/data` and `rag/data/index` are available in the runtime container/host |
| Groq API failure | missing key, wrong URL, or model mismatch | set `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` correctly |
| missing API key | optional generation path is enabled without config | either set Groq variables or avoid `?generate=true` |
| `grounded=false` | evidence weak or irrelevant | check the user question and the retrieved top chunk; the system is intentionally conservative |
| slow generation | optional Groq call is being used | use the default retrieval endpoint for latency-sensitive workloads |
| Windows encoding issues | non-UTF-8 text or shell locale issues | use UTF-8 output and run from a shell with proper encoding |

## 22) ⚡ 5-Minute Teammate Integration

1. Clone the repository
2. Checkout the branch you want to use
3. `cd rag`
4. `python -m venv .venv`
5. Activate the environment
6. `pip install -r requirements.txt`
7. `Copy-Item .env.example .env` or `cp .env.example .env`
8. Fill the required values in `.env`
9. Start the service: `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`
10. Test `GET /health`
11. Send `POST /rag/query` with a question in JSON
12. Use `result.answer` in the main app or frontend

## 23) Production integration checklist

Use this list before launch:

[ ] RAG service starts
[ ] /health returns ok
[ ] /rag/query works
[ ] frontend/backend can reach RAG
[ ] CORS configured
[ ] production URL configured
[ ] dataset/index available
[ ] .env configured
[ ] secrets not committed
[ ] STT produces text
[ ] text reaches /rag/query
[ ] answer reaches frontend
[ ] grounded=false handled
[ ] end-to-end voice test passed

## 24) Final API reference

### Base URL

```text
http://YOUR_RAG_HOST:8000
```

### GET /health

```bash
curl http://YOUR_RAG_HOST:8000/health
```

### POST /rag/query

```bash
curl -X POST http://YOUR_RAG_HOST:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"भारत की राजधानी क्या है?"}'
```

### Request body

```json
{
  "query": "भारत की राजधानी क्या है?"
}
```

### Successful response

```json
{
  "answer": "नई दिल्ली भारत की राजधानी है।",
  "grounded": true,
  "confidence": 0.8,
  "sources": [
    {
      "query_id": "123",
      "chunk_id": "1-0-0",
      "text": "नई दिल्ली भारत की राजधानी है।",
      "language": "hi",
      "chunking_strategy": "sentence",
      "score": 0.8
    }
  ],
  "latency_ms": 12.5
}
```

### Refusal response

```json
{
  "answer": "संदर्भ में पर्याप्त जानकारी नहीं है",
  "grounded": false,
  "confidence": 0.0,
  "sources": [],
  "latency_ms": 8.1
}
```

### Optional Groq mode

```bash
curl -X POST "http://YOUR_RAG_HOST:8000/rag/query?generate=true" \
  -H "Content-Type: application/json" \
  -d '{"query":"भारत की राजधानी क्या है?"}'
```

## Final notes

This service is a clean, isolated text-to-RAG API. It does not handle audio or UI. The teammate only needs to provide transcribed text and call `POST /rag/query`.

The default path is the fast retrieval answer path. The Groq path is optional and slower. The service is designed to refuse unsupported questions rather than fabricate an answer.

If the teammate has the repo and a local Python environment, the service is ready to run without needing any additional project architecture changes.
