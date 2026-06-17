from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    TITLE = "title"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    LIST = "list"
    HEADER = "header"
    FOOTER = "footer"
    OTHER = "other"


class TableData(BaseModel):
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
    page_no: int
    elements: list[PageElement] = Field(default_factory=list)
    is_scanned: bool = False
    char_count: int = 0


class ParsedDocument(BaseModel):
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
    doc_id: str
    source_filename: str
    source_hash: str
    normalized_uri: Optional[str] = None
    page_count: int = 0
    is_scanned: bool = False
    language: str = "zh"


class DocumentRecord(BaseModel):
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
