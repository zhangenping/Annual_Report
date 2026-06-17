"""Type-aware chunking with parent-child support."""

from __future__ import annotations

from annual_report_rag.schemas.chunk import Chunk, ChunkMetadata, ChunkType
from annual_report_rag.schemas.document import ElementType, ParsedDocument
from annual_report_rag.utils import estimate_tokens, new_id, token_split


class ChunkBuilder:
    def __init__(
        self,
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        table_max_rows: int = 40,
        parse_version: str = "v1.0.0",
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.table_max_rows = table_max_rows
        self.parse_version = parse_version

    def build(self, parsed: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        section_stack: list[str] = []

        for page in parsed.pages:
            for element in sorted(page.elements, key=lambda e: e.reading_order):
                if element.type == ElementType.TITLE:
                    section_stack = _update_sections(section_stack, element.text)
                    continue
                if element.type in {ElementType.HEADER, ElementType.FOOTER}:
                    continue

                base_meta = ChunkMetadata(
                    company_id=parsed.company_id,
                    company_name=parsed.company_name,
                    fiscal_year=parsed.fiscal_year,
                    report_type=parsed.report_type,
                    language=parsed.language,
                    section=section_stack[-1] if section_stack else None,
                    section_path=list(section_stack),
                    page_start=element.page_no,
                    page_end=element.page_no,
                    source_file=parsed.source_filename,
                    source_hash=parsed.source_hash,
                    parse_pipeline_version=self.parse_version,
                    element_id=element.element_id,
                )

                if element.type == ElementType.TABLE and element.table:
                    chunks.extend(self._chunk_table(parsed, element, base_meta))
                elif element.type == ElementType.FIGURE:
                    chunks.append(self._chunk_figure(parsed, element, base_meta))
                elif element.text.strip():
                    chunks.extend(self._chunk_text(parsed, element, base_meta))

        return chunks

    def _chunk_text(self, parsed: ParsedDocument, element, base_meta: ChunkMetadata) -> list[Chunk]:
        parts = token_split(element.text, self.max_tokens, self.overlap_tokens)
        if not parts:
            return []
        parent_id = new_id()
        parent_text = element.text
        parent = Chunk(
            chunk_id=parent_id,
            doc_id=parsed.doc_id,
            chunk_type=ChunkType.TEXT,
            content_text=parent_text,
            content_md=parent_text,
            metadata=base_meta,
            token_count=estimate_tokens(parent_text),
        )
        result = [parent]
        if len(parts) == 1:
            return result

        for part in parts:
            result.append(
                Chunk(
                    chunk_id=new_id(),
                    doc_id=parsed.doc_id,
                    parent_chunk_id=parent_id,
                    chunk_type=ChunkType.TEXT,
                    content_text=part,
                    content_md=part,
                    metadata=base_meta,
                    token_count=estimate_tokens(part),
                )
            )
        return result

    def _chunk_table(self, parsed: ParsedDocument, element, base_meta: ChunkMetadata) -> list[Chunk]:
        table = element.table
        assert table is not None
        parent_id = new_id()
        full_md = table.markdown or element.text
        table_json = table.model_dump(mode="json")

        parent = Chunk(
            chunk_id=parent_id,
            doc_id=parsed.doc_id,
            chunk_type=ChunkType.TABLE,
            content_text=full_md,
            content_md=full_md,
            table_json=table_json,
            metadata=base_meta,
            token_count=estimate_tokens(full_md),
        )
        chunks = [parent]

        rows = table.rows
        if len(rows) <= self.table_max_rows:
            return chunks

        headers = table.headers[-1] if table.headers else []
        for i in range(0, len(rows), self.table_max_rows):
            group = rows[i : i + self.table_max_rows]
            from annual_report_rag.utils import table_to_markdown

            part_md = table_to_markdown([headers], group)
            caption = table.caption or base_meta.section or "表格"
            content = f"{caption}（第{i+1}-{i+len(group)}行）\n{part_md}"
            chunks.append(
                Chunk(
                    chunk_id=new_id(),
                    doc_id=parsed.doc_id,
                    parent_chunk_id=parent_id,
                    chunk_type=ChunkType.TABLE,
                    content_text=content,
                    content_md=part_md,
                    table_json={
                        **table_json,
                        "rows": group,
                        "row_offset": i,
                    },
                    metadata=base_meta,
                    token_count=estimate_tokens(content),
                )
            )
        return chunks

    def _chunk_figure(self, parsed: ParsedDocument, element, base_meta: ChunkMetadata) -> Chunk:
        caption = element.figure_caption or element.text
        summary = f"【图表】{caption}"
        if element.text and element.text != caption:
            summary += f"\n说明：{element.text}"
        return Chunk(
            chunk_id=new_id(),
            doc_id=parsed.doc_id,
            chunk_type=ChunkType.FIGURE,
            content_text=summary,
            content_md=summary,
            figure_uri=element.figure_uri,
            metadata=base_meta,
            token_count=estimate_tokens(summary),
        )


def _update_sections(stack: list[str], title: str) -> list[str]:
    if "第" in title and "节" in title[:8]:
        return [title]
    if len(stack) >= 3:
        return stack[:2] + [title]
    return stack + [title]
