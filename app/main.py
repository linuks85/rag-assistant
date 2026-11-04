"""FastAPI service for the document assistant."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from corpus import SEED_CORPUS
from app.llm_client import LLMClient
from app.rag import rag_answer_with_guard
from app.retriever import Retriever

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["retriever"] = Retriever(path="./chroma_rag")
    state["retriever"].index(SEED_CORPUS)
    state["generator"] = LLMClient()
    try:
        yield
    finally:
        state.clear()


app = FastAPI(title="RAG Document Assistant", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(5, ge=1, le=20)


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_ids: list[str]
    top_similarity: float
    guarded: bool


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        result = rag_answer_with_guard(
            state["retriever"], state["generator"], request.question, k=request.k
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RAG query failed: {exc}") from exc
    return QueryResponse(**result)
