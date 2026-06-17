"""
FastAPI 服务：对外暴露检索与 Agent 问答能力。

端点：
  GET  /health                  健康检查
  POST /api/v1/search           混合检索（无需 LLM）
  POST /api/v1/ask              Agent 问答（需 OPENAI_API_KEY）
  POST /api/v1/ingest/rebuild-index  重建索引（不重新解析）

单例懒加载 HybridSearch / Agent，避免重复加载 Embedding 模型。
"""

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
    """混合检索接口，返回带 citation 的 Chunk 列表。"""
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
    """Agent 问答：多步检索 + 工具调用 + 引用溯源。"""
    try:
        return get_agent().ask(req.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/ingest/rebuild-index")
def rebuild_index() -> dict[str, Any]:
    """解析完成后单独重建索引，并刷新 API 内存缓存。"""
    pipeline = IngestPipeline()
    stats = pipeline.rebuild_indexes()
    get_search().refresh()
    return {"index": stats}
