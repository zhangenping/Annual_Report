"""
文档解析阶段的 Pydantic 数据模型。

层级关系：
  ParsedDocument → PageModel[] → PageElement[]
                                    ├── TableData（表格结构化）
                                    └── figure_uri（图片路径）

DocumentRecord：入库登记表，存于 data/documents.json
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    """页面元素类型，驱动切片器的分路逻辑。"""

    TITLE = "title"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    LIST = "list"
    HEADER = "header"
    FOOTER = "footer"
    OTHER = "other"


class TableData(BaseModel):
    """
    表格结构化表示（机器可读）。

    与 markdown 字段双写：前者供 Agent 精确查数，后者供检索与展示。
    """

    table_id: str
    caption: Optional[str] = None
    unit: Optional[str] = None
    currency: Optional[str] = None
    headers: list[list[str]] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    markdown: Optional[str] = None
    source_page: Optional[int] = None
    bbox: Optional[list[float]] = None


class PageElement(BaseModel):
    """单页内的一个版面元素。"""

    element_id: str
    type: ElementType
    text: str = ""
    reading_order: int = 0
    bbox: Optional[list[float]] = None
    page_no: int = 1
    table: Optional[TableData] = None
    figure_uri: Optional[str] = None
    figure_caption: Optional[str] = None
    children: list["PageElement"] = Field(default_factory=list)


class PageModel(BaseModel):
    """单页及其元素列表。"""

    page_no: int
    elements: list[PageElement] = Field(default_factory=list)
    is_scanned: bool = False  # 扫描页标记，后续可触发 OCR
    char_count: int = 0


class ParsedDocument(BaseModel):
    """解析管线输出：一份年报的结构化中间表示。"""

    doc_id: str
    source_filename: str
    source_hash: str
    company_id: Optional[str] = None
    company_name: Optional[str] = None
    fiscal_year: Optional[int] = None
    report_type: str = "annual"
    language: str = "zh"
    is_scanned: bool = False
    page_count: int = 0
    parse_engine: str = "pymupdf"
    parse_version: str = "v1.0.0"
    pages: list[PageModel] = Field(default_factory=list)
    title: Optional[str] = None


class DocumentManifest(BaseModel):
    """格式归一化阶段的清单（预留，用于记录转换后 PDF 信息）。"""

    doc_id: str
    source_filename: str
    source_hash: str
    normalized_uri: Optional[str] = None
    page_count: int = 0
    is_scanned: bool = False
    language: str = "zh"


class DocumentRecord(BaseModel):
    """文档入库登记条目，写入 data/documents.json。"""

    doc_id: str
    company_id: str
    company_name: str
    fiscal_year: int
    report_type: str = "annual"
    source_filename: str
    source_hash: str
    storage_uri: str
    parse_status: str = "pending"
    parse_version: Optional[str] = None
    chunk_count: int = 0
