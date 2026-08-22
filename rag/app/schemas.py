from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)

    @field_validator("query")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be empty")
        return value


class Source(BaseModel):
    query_id: str
    chunk_id: str
    text: str
    language: str
    chunking_strategy: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[Source]
    latency_ms: float = Field(ge=0.0)


class GeneratedAnswer(BaseModel):
    answer: str
    grounded: bool
