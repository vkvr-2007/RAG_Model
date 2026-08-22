# HH Goa 2026 Shortlisting Task 2: Voice-Enabled RAG Model

This repository already contains the RAG service scaffold and a real local Hindi MSMARCO-XI parquet dataset. The work here focuses on the RAG component only; speech-to-text and frontend code remain out of scope.

## Real local dataset in this workspace

The real local dataset is already present in the project:

- `data/hintrain.parquet` — full local Hindi MSMARCO-XI shard (778,638 rows)
- `data/hindi_subset.parquet` — real subset extracted from that shard (5,000 rows)

The project uses the authentic local MSMARCO-XI schema, including fields such as:

- `query`
- `Answer`
- `query_id`
- `query_type`
- `passages.is_selected`
- `passages.English_passages`
- `passages.Translated_passages`
- `Eng_Query`
- `Eng_Answer`

The active prototype uses the real subset `data/hindi_subset.parquet` for offline indexing.

## RAG API

By default, the API uses the fast grounded retrieval path and does not call the external LLM. If you explicitly want the slower optional Groq generation path, send `?generate=true`.

Start the service:

```powershell
cd "c:\Users\REHAN\OneDrive\Desktop\rag"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

Example query (default fast retrieval path):

```powershell
curl -X POST http://127.0.0.1:8000/rag/query -H "Content-Type: application/json" -d "{\"query\":\"भारत की राजधानी क्या है?\"}"
```

Optional slower generation mode:

```powershell
curl -X POST "http://127.0.0.1:8000/rag/query?generate=true" -H "Content-Type: application/json" -d "{\"query\":\"भारत की राजधानी क्या है?\"}"
```

Response shape:

```json
{
  "answer": "...",
  "grounded": true,
  "confidence": 0.0,
  "sources": [],
  "latency_ms": 0
}
```

For unsupported or low-confidence queries, the API returns:

```json
{
  "answer": "संदर्भ में पर्याप्त जानकारी नहीं है",
  "grounded": false,
  "confidence": 0.0,
  "sources": [],
  "latency_ms": 0
}
```

## Frontend integration example

```javascript
const response = await fetch("http://YOUR_SERVER:8000/rag/query", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    query: userTranscribedText
  })
});

const data = await response.json();
console.log(data.answer);
```

The teammate only needs to send the transcribed question text.

## CORS

The API allows common local frontend origins through the `CORS_ALLOWED_ORIGINS` environment variable. Configure it as a comma-separated list. Example:

```env
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173
```

## Build or rebuild the index

Use the real local subset and offline chunking:

```powershell
cd "c:\Users\REHAN\OneDrive\Desktop\rag"
python -m app.indexer --input-parquet data/hindi_subset.parquet --limit 5000 --strategy sentence --embedding-backend hash --output-dir data/index
```

If the environment is able to load a sentence-transformers model, the app will use it automatically. If not, the embedder falls back to the hash-based deterministic local embedding backend so the index still builds reliably in constrained environments.

## Environment variables

Copy `.env.example` to `.env` and set your required values:

```powershell
Copy-Item .env.example .env
```

Example:

```env
RAG_EMBEDDING_BACKEND=hash
RAG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RAG_USE_BM25=true
RAG_TOP_K=4
RAG_MIN_RETRIEVAL_SCORE=0.05
RAG_STRICT_EXTRACTIVE=true
LLM_BASE_URL=https://your-provider.example/v1
LLM_API_KEY=your-api-key
LLM_MODEL=your-model
LLM_TIMEOUT_SECONDS=12
```

## Chunking strategies

The project supports three offline, configurable chunking strategies:

- `passage`
- `sentence`
- `recursive`

Each chunk preserves metadata including:

- `chunk_id`
- `query_id`
- `source_passage`
- `language`
- `chunking_strategy`

## Retrieval and guardrails

The service performs offline preprocessing and indexing, then loads the already-built FAISS index. Retrieval uses vector search plus BM25 reciprocal-rank fusion. Low-confidence or ungrounded answers return a safe refusal rather than a hallucinated answer.

Guardrails include:

- empty query rejection
- malformed data rejection
- low-confidence refusal
- off-topic/unanswerable refusal
- no fabricated answer without evidence
- generator fallback on failure

## Benchmarking

After starting the API, use an actual local query file:

```powershell
python scripts\benchmark.py --queries-file data\benchmark_queries_100.txt
```

The benchmark reports measured P50, P70, and P100 values from real requests.

## Tests

Run the real project tests for the RAG component:

```powershell
python -m pytest tests -q
```

## Notes

- The environment does not expose browser-based HF dataset access, so this implementation uses the real local parquet files already present in the workspace.
- The app does not include STT or frontend handling by design.
- The generated index is saved under `data/index` and is reused on startup instead of being rebuilt per request.
