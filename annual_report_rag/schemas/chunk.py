from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    LIST = "list"


class ChunkMetadata(BaseModel):
    company_id: Optional[str] = None
    company_name: Optional[str] = None
    fiscal_year: Optional[int] = None
    report_type: str = "annual"
    language: str = "zh"
    section: Optional[str] = None
    section_path: list[str] = Field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    source_file: str
    source_hash: str
    parse_pipeline_version: str = "v1.0.0"
    element_id: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    parent_chunk_id: Optional[str] = None
    chunk_type: ChunkType
    content_text: str
    content_md: Optional[str] = None
    table_json: Optional[dict[str, Any]] = None
    figure_uri: Optional[str] = None
    metadata: ChunkMetadata
    token_count: int = 0
    embedding_id: Optional[str] = None

    def citation(self) -> str:
        meta = self.metadata
        parts = [
            meta.company_name or meta.company_id or "未知公司",
            f"{meta.fiscal_year}年" if meta.fiscal_year else "",
            meta.section or "",
        ]
        if meta.page_start:
            parts.append(f"第{meta.page_start}页")
        return "，".join(p for p in parts if p)

    def embedding_text(self) -> str:
        prefix_parts = self.metadata.section_path or []
        if self.metadata.company_name:
            prefix_parts = [self.metadata.company_name] + prefix_parts
        prefix = " > ".join(prefix_parts)
        if prefix:
            return f"[{prefix}]\n{self.content_text}"
        return self.content_text
