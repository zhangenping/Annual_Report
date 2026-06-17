"""
检索与索引阶段的 Chunk 数据模型。

Chunk 是 RAG 知识库的最小检索单元，携带：
  - content_text：检索与 Embedding 主文本
  - table_json / figure_uri：多模态扩展字段
  - metadata：公司、年份、章节、页码等过滤维度
  - parent_chunk_id：Parent-Child 关联
"""

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
    """切片元数据：检索过滤与引用溯源的核心字段。"""

    company_id: Optional[str] = None
    company_name: Optional[str] = None
    fiscal_year: Optional[int] = None
    report_type: str = "annual"
    language: str = "zh"
    section: Optional[str] = None
    section_path: list[str] = Field(default_factory=list)  # 章节面包屑
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    source_file: str
    source_hash: str
    parse_pipeline_version: str = "v1.0.0"
    element_id: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """知识库切片实体。"""

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
        """生成人类可读的引用字符串，如「贵州茅台，2023年，第三节，第42页」。"""
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
        """
        送入 Embedding 模型的文本（Contextual Retrieval）。

        在正文前拼接章节路径，提升「第三节 研发费用」类查询的召回率。
        """
        prefix_parts = self.metadata.section_path or []
        if self.metadata.company_name:
            prefix_parts = [self.metadata.company_name] + prefix_parts
        prefix = " > ".join(prefix_parts)
        if prefix:
            return f"[{prefix}]\n{self.content_text}"
        return self.content_text
