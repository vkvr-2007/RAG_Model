from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    index_dir: Path = Field(default=Path("data/index"), alias="RAG_INDEX_DIR")
    embedding_model: str = Field(default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", alias="RAG_EMBEDDING_MODEL")
    embedding_backend: str = Field(default="sentence_transformers", alias="RAG_EMBEDDING_BACKEND")
    use_bm25: bool = Field(default=True, alias="RAG_USE_BM25")
    top_k: int = Field(default=4, alias="RAG_TOP_K")
    min_retrieval_score: float = Field(default=0.05, alias="RAG_MIN_RETRIEVAL_SCORE")
    llm_base_url: str | None = Field(default=None, alias="LLM_BASE_URL")
    llm_api_key: str | None = Field(default=None, alias="LLM_API_KEY")
    llm_model: str | None = Field(default=None, alias="LLM_MODEL")
    llm_timeout_seconds: float = Field(default=12.0, alias="LLM_TIMEOUT_SECONDS")
    strict_extractive: bool = Field(default=True, alias="RAG_STRICT_EXTRACTIVE")
    cors_allowed_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080",
        alias="CORS_ALLOWED_ORIGINS",
    )


settings = Settings()
