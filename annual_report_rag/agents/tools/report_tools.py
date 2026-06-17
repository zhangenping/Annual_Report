"""Agent tools for annual report QA."""

from __future__ import annotations

import json
import re
from typing import Any

from annual_report_rag.retrieval import HybridSearch, SearchFilters
from annual_report_rag.storage import LocalStore


class ReportTools:
    def __init__(self, search: HybridSearch | None = None) -> None:
        self.search = search or HybridSearch()
        self.store = self.search.store

    def search_annual_report(
        self,
        query: str,
        company_id: str | None = None,
        fiscal_years: list[int] | None = None,
        chunk_types: list[str] | None = None,
        top_k: int = 8,
    ) -> str:
        filters = SearchFilters(
            company_id=company_id,
            fiscal_years=fiscal_years or [],
            chunk_types=chunk_types or [],
        )
        hits = self.search.search(query, filters=filters, top_k=top_k, rerank=True)
        return json.dumps(hits, ensure_ascii=False, indent=2)

    def get_chunk_detail(self, chunk_id: str) -> str:
        detail = self.search.get_chunk(chunk_id, include_parent=True)
        return json.dumps(detail or {}, ensure_ascii=False, indent=2)

    def query_financial_table(self, chunk_id: str, row_keyword: str) -> str:
        detail = self.search.get_chunk(chunk_id, include_parent=False)
        if not detail:
            return json.dumps({"error": "chunk not found"})
        chunk = detail["chunk"]
        table = chunk.get("table_json") or {}
        rows = table.get("rows", [])
        matched = [row for row in rows if row_keyword in "".join(map(str, row))]
        return json.dumps(
            {"chunk_id": chunk_id, "matches": matched, "headers": table.get("headers", [])},
            ensure_ascii=False,
            indent=2,
        )

    def list_sections(self, company_id: str | None = None, fiscal_year: int | None = None) -> str:
        sections: set[str] = set()
        for chunk in self.store.load_all_chunks():
            meta = chunk.metadata
            if company_id and meta.company_id != company_id:
                continue
            if fiscal_year and meta.fiscal_year != fiscal_year:
                continue
            if meta.section:
                sections.add(meta.section)
        return json.dumps(sorted(sections), ensure_ascii=False)

    def compare_metrics(self, query: str, fiscal_years: list[int]) -> str:
        all_hits = []
        for year in fiscal_years:
            filters = SearchFilters(fiscal_years=[year])
            hits = self.search.search(query, filters=filters, top_k=5, rerank=True)
            all_hits.append({"year": year, "hits": hits})
        return json.dumps(all_hits, ensure_ascii=False, indent=2)

    @staticmethod
    def calculator(expression: str) -> str:
        allowed = re.compile(r"^[\d\s+\-*/().%]+$")
        if not allowed.match(expression):
            return json.dumps({"error": "invalid expression"})
        try:
            value = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
            return json.dumps({"result": value})
        except Exception as exc:
            return json.dumps({"error": str(exc)})


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_annual_report",
            "description": "混合检索年报知识库，返回相关切片与引用",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "company_id": {"type": "string"},
                    "fiscal_years": {"type": "array", "items": {"type": "integer"}},
                    "chunk_types": {"type": "array", "items": {"type": "string"}},
                    "top_k": {"type": "integer", "default": 8},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk_detail",
            "description": "按 chunk_id 获取切片及父级上下文",
            "parameters": {
                "type": "object",
                "properties": {"chunk_id": {"type": "string"}},
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_financial_table",
            "description": "在表格切片中按行关键词查询",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "row_keyword": {"type": "string"},
                },
                "required": ["chunk_id", "row_keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_metrics",
            "description": "按多个财年分别检索并对比",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "fiscal_years": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["query", "fiscal_years"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "安全计算简单算术表达式",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]
