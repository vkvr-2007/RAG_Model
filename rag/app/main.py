from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.embeddings import build_embedder
from app.generation import Generator
from app.retrieval import Retriever
from app.schemas import QueryRequest, QueryResponse
from app.service import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if not (settings.index_dir / "vectors.faiss").exists():
            raise FileNotFoundError(f"Missing index at {settings.index_dir}; run python -m app.indexer first.")
        embedder = build_embedder(settings.embedding_backend, settings.embedding_model)
        retriever = Retriever(settings.index_dir, embedder, settings.use_bm25)
        app.state.rag = RAGService(retriever, Generator(settings.llm_base_url, settings.llm_api_key, settings.llm_model, settings.llm_timeout_seconds), settings.top_k, settings.min_retrieval_score, True)
    except Exception as error:
        app.state.startup_error = str(error)
    yield


app = FastAPI(title="Hindi RAG Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_allowed_origins.split(",") if origin.strip()],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=False,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    normalized = []
    for error in exc.errors():
        safe = dict(error)
        ctx = safe.get("ctx")
        if isinstance(ctx, dict):
            safe["ctx"] = {key: str(value) if isinstance(value, Exception) else value for key, value in ctx.items()}
        normalized.append(safe)
    return JSONResponse(status_code=422, content={"detail": "Invalid request payload", "errors": normalized})


@app.get("/health")
async def health(request: Request):
    error = getattr(request.app.state, "startup_error", None)
    return {
        "status": "ok" if not error else "degraded",
        "index_loaded": hasattr(request.app.state, "rag"),
        "embedding_backend": settings.embedding_backend,
        "retrieval_ready": hasattr(request.app.state, "rag"),
        "detail": error,
    }


@app.post("/rag/query", response_model=QueryResponse)
async def rag_query(payload: QueryRequest, request: Request, generate: bool = False) -> QueryResponse:
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    service = getattr(request.app.state, "rag", None)
    if not service:
        raise HTTPException(status_code=503, detail="RAG index is unavailable; run the offline indexer first.")
    try:
        return await service.query(payload.query, generate=generate)
    except TypeError as exc:
        if "generate" not in str(exc):
            raise
        return await service.query(payload.query)
