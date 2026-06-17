"""PDF layout parsing with Docling (optional) and PyMuPDF fallback."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import fitz

from annual_report_rag.schemas.document import (
    ElementType,
    PageElement,
    PageModel,
    ParsedDocument,
    TableData,
)
from annual_report_rag.utils import (
    clean_text,
    file_sha256,
    guess_company_from_text,
    new_id,
    table_to_markdown,
)

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    @abstractmethod
    def parse(
        self,
        pdf_path: Path,
        *,
        parse_version: str,
        extract_images: bool = True,
        figures_dir: Path | None = None,
        doc_id: str | None = None,
    ) -> ParsedDocument:
        raise NotImplementedError


class PyMuPDFParser(BaseParser):
    """Fallback parser using PyMuPDF blocks and pdfplumber tables."""

    def parse(
        self,
        pdf_path: Path,
        *,
        parse_version: str,
        extract_images: bool = True,
        figures_dir: Path | None = None,
        doc_id: str | None = None,
    ) -> ParsedDocument:
        doc_id = doc_id or new_id()
        source_hash = file_sha256(pdf_path)
        doc = fitz.open(pdf_path)

        pages: list[PageModel] = []
        all_text_parts: list[str] = []
        section_stack: list[str] = []

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_no = page_index + 1
            blocks = page.get_text("dict")["blocks"]
            char_count = len(page.get_text())
            page_elements: list[PageElement] = []
            order = 0

            for block in blocks:
                if block.get("type") != 0:
                    continue
                lines = []
                for line in block.get("lines", []):
                    spans = "".join(span.get("text", "") for span in line.get("spans", []))
                    lines.append(spans)
                text = clean_text(" ".join(lines))
                if not text or len(text) < 2:
                    continue

                bbox = block.get("bbox")
                element_type = _classify_text(text)
                if element_type == ElementType.TITLE:
                    section_stack = _update_section_stack(section_stack, text)

                page_elements.append(
                    PageElement(
                        element_id=f"p{page_no}_e{order}",
                        type=element_type,
                        text=text,
                        reading_order=order,
                        bbox=list(bbox) if bbox else None,
                        page_no=page_no,
                    )
                )
                all_text_parts.append(text)
                order += 1

            # Tables via pdfplumber
            try:
                import pdfplumber

                with pdfplumber.open(pdf_path) as pdf:
                    if page_index < len(pdf.pages):
                        plumber_page = pdf.pages[page_index]
                        for t_idx, table in enumerate(plumber_page.extract_tables() or []):
                            if not table or len(table) < 2:
                                continue
                            headers = [[str(c or "") for c in table[0]]]
                            rows = [[str(c or "") for c in row] for row in table[1:]]
                            markdown = table_to_markdown(headers, rows)
                            table_id = f"t_{page_no}_{t_idx}"
                            page_elements.append(
                                PageElement(
                                    element_id=f"p{page_no}_t{t_idx}",
                                    type=ElementType.TABLE,
                                    text=markdown,
                                    reading_order=order,
                                    page_no=page_no,
                                    table=TableData(
                                        table_id=table_id,
                                        headers=headers,
                                        rows=[[str(c or "") for c in row] for row in rows],
                                        markdown=markdown,
                                        source_page=page_no,
                                    ),
                                )
                            )
                            order += 1
            except Exception as exc:
                logger.warning("Table extraction failed on page %s: %s", page_no, exc)

            # Images
            if extract_images and figures_dir:
                figures_dir.mkdir(parents=True, exist_ok=True)
                for img_idx, img in enumerate(page.get_images(full=True)):
                    try:
                        xref = img[0]
                        base = doc.extract_image(xref)
                        if base["width"] < 80 or base["height"] < 80:
                            continue
                        ext = base.get("ext", "png")
                        fname = f"p{page_no}_img{img_idx}.{ext}"
                        out_path = figures_dir / fname
                        out_path.write_bytes(base["image"])
                        page_elements.append(
                            PageElement(
                                element_id=f"p{page_no}_f{img_idx}",
                                type=ElementType.FIGURE,
                                text=f"[图片 第{page_no}页]",
                                reading_order=order,
                                page_no=page_no,
                                figure_uri=str(out_path),
                                figure_caption=f"第{page_no}页 图{img_idx + 1}",
                            )
                        )
                        order += 1
                    except Exception as exc:
                        logger.debug("Skip image %s on page %s: %s", img_idx, page_no, exc)

            page_elements.sort(key=lambda e: e.reading_order)
            pages.append(
                PageModel(
                    page_no=page_no,
                    elements=page_elements,
                    is_scanned=char_count < 50,
                    char_count=char_count,
                )
            )

        doc.close()
        joined = "\n".join(all_text_parts)
        company_name, company_id, fiscal_year = guess_company_from_text(joined)

        return ParsedDocument(
            doc_id=doc_id,
            source_filename=pdf_path.name,
            source_hash=source_hash,
            company_id=company_id,
            company_name=company_name,
            fiscal_year=fiscal_year,
            is_scanned=sum(1 for p in pages if p.is_scanned) > len(pages) // 2,
            page_count=len(pages),
            parse_engine="pymupdf",
            parse_version=parse_version,
            pages=pages,
            title=company_name,
        )


class DoclingParser(BaseParser):
    """Docling adapter when installed."""

    def parse(
        self,
        pdf_path: Path,
        *,
        parse_version: str,
        extract_images: bool = True,
        figures_dir: Path | None = None,
        doc_id: str | None = None,
    ) -> ParsedDocument:
        from docling.document_converter import DocumentConverter

        doc_id = doc_id or new_id()
        source_hash = file_sha256(pdf_path)
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        dl_doc = result.document

        pages_map: dict[int, list[PageElement]] = {}
        order_counters: dict[int, int] = {}

        def page_order(page_no: int) -> int:
            order_counters[page_no] = order_counters.get(page_no, 0)
            val = order_counters[page_no]
            order_counters[page_no] += 1
            return val

        for item, _level in dl_doc.iterate_items():
            label = getattr(item, "label", None)
            text = clean_text(getattr(item, "text", "") or "")
            page_no = 1
            prov = getattr(item, "prov", None)
            if prov:
                page_no = getattr(prov[0], "page_no", 1)

            element_type = ElementType.PARAGRAPH
            if label:
                label_s = str(label).lower()
                if "title" in label_s or "heading" in label_s:
                    element_type = ElementType.TITLE
                elif "table" in label_s:
                    element_type = ElementType.TABLE
                elif "picture" in label_s or "figure" in label_s:
                    element_type = ElementType.FIGURE
                elif "list" in label_s:
                    element_type = ElementType.LIST

            table_data = None
            if element_type == ElementType.TABLE and hasattr(item, "export_to_dataframe"):
                try:
                    df = item.export_to_dataframe()
                    headers = [list(df.columns.astype(str))]
                    rows = df.astype(str).values.tolist()
                    markdown = table_to_markdown(headers, rows)
                    table_data = TableData(
                        table_id=f"t_{page_no}_{page_order(page_no)}",
                        headers=headers,
                        rows=rows,
                        markdown=markdown,
                        source_page=page_no,
                    )
                    text = markdown
                except Exception:
                    pass

            if not text and element_type != ElementType.FIGURE:
                continue

            pages_map.setdefault(page_no, []).append(
                PageElement(
                    element_id=f"p{page_no}_e{page_order(page_no)}",
                    type=element_type,
                    text=text or "[图表]",
                    reading_order=page_order(page_no),
                    page_no=page_no,
                    table=table_data,
                )
            )

        pages: list[PageModel] = []
        all_text: list[str] = []
        for page_no in sorted(pages_map):
            elements = sorted(pages_map[page_no], key=lambda e: e.reading_order)
            text_joined = " ".join(e.text for e in elements)
            all_text.append(text_joined)
            pages.append(
                PageModel(
                    page_no=page_no,
                    elements=elements,
                    char_count=len(text_joined),
                    is_scanned=len(text_joined) < 50,
                )
            )

        joined = "\n".join(all_text)
        company_name, company_id, fiscal_year = guess_company_from_text(joined)

        return ParsedDocument(
            doc_id=doc_id,
            source_filename=pdf_path.name,
            source_hash=source_hash,
            company_id=company_id,
            company_name=company_name,
            fiscal_year=fiscal_year,
            is_scanned=False,
            page_count=len(pages) or 1,
            parse_engine="docling",
            parse_version=parse_version,
            pages=pages,
            title=company_name,
        )


def get_parser(engine: str = "auto") -> BaseParser:
    if engine == "pymupdf":
        return PyMuPDFParser()
    if engine == "docling":
        return DoclingParser()
    try:
        import docling  # noqa: F401

        return DoclingParser()
    except ImportError:
        logger.info("Docling not installed, using PyMuPDF parser.")
        return PyMuPDFParser()


def _classify_text(text: str) -> ElementType:
    if len(text) <= 40 and not text.endswith("。"):
        if any(k in text for k in ("报告", "摘要", "目录", "第一节", "第二节", "第三节", "附注")):
            return ElementType.TITLE
        if re_short_title(text):
            return ElementType.TITLE
    if text.startswith("第") and "节" in text[:10]:
        return ElementType.TITLE
    return ElementType.PARAGRAPH


def re_short_title(text: str) -> bool:
    import re

    return bool(re.match(r"^[\u4e00-\u9fff\d、．.\s]{2,30}$", text))


def _update_section_stack(stack: list[str], title: str) -> list[str]:
    depth = 1
    if "第" in title and "节" in title:
        depth = 1
    elif len(title) <= 15:
        depth = min(len(stack) + 1, 3)
    new_stack = stack[: depth - 1]
    new_stack.append(title)
    return new_stack[:4]
