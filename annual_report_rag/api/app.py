"""FastAPI endpoints for search and agent QA."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from annual_report_rag.agents.graph import AnnualReportAgent
from annual_report_rag.pipelines.ingest.pipeline import IngestPipeline
from annual_report_rag.retrieval import HybridSearch, SearchFilters

app = FastAPI(
    title="Annual Report RAG API",
    description="企业年报知识库检索与 Agent 问答 API",
    version="0.1.0",
)

_search: HybridSearch | None = None
_agent: AnnualReportAgent | None = None


def get_search() -> HybridSearch:
    global _search
    if _search is None:
        _search = HybridSearch()
    return _search


def get_agent() -> AnnualReportAgent:
    global _agent
    if _agent is None:
        _agent = AnnualReportAgent(get_search())
    return _agent


class SearchRequest(BaseModel):
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = 10
    rerank: bool = True


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/search")
def search(req: SearchRequest) -> dict[str, Any]:
    filters = SearchFilters(
        company_id=req.filters.get("company_id"),
        fiscal_years=req.filters.get("fiscal_years", []),
        chunk_types=req.filters.get("chunk_types", []),
        section_contains=req.filters.get("section_contains"),
    )
    chunks = get_search().search(
        req.query,
        filters=filters,
        top_k=req.top_k,
        rerank=req.rerank,
    )
    return {"chunks": chunks}


@app.post("/api/v1/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    try:
        return get_agent().ask(req.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/ingest/rebuild-index")
def rebuild_index() -> dict[str, Any]:
    pipeline = IngestPipeline()
    stats = pipeline.rebuild_indexes()
    get_search().refresh()
    return {"index": stats}
