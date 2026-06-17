"""End-to-end ingestion orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

from annual_report_rag.config import load_yaml_config, resolve_path
from annual_report_rag.pipelines.chunk.builder import ChunkBuilder
from annual_report_rag.pipelines.index.builder import BM25Index, VectorIndex
from annual_report_rag.pipelines.normalize.converter import normalize_to_pdf
from annual_report_rag.pipelines.parse.pdf_parser import get_parser
from annual_report_rag.schemas.document import DocumentRecord
from annual_report_rag.storage import LocalStore
from annual_report_rag.utils import file_sha256, new_id

logger = logging.getLogger(__name__)


class IngestPipeline:
    def __init__(self) -> None:
        self.pipeline_cfg = load_yaml_config("pipelines.yaml")
        self.store = LocalStore(self.pipeline_cfg["paths"]["data_dir"])
        paths = self.pipeline_cfg["paths"]
        self.raw_dir = resolve_path(paths["raw_dir"])
        self.parse_cfg = self.pipeline_cfg["parse"]
        self.chunk_cfg = self.pipeline_cfg["chunk"]
        self.index_cfg = self.pipeline_cfg["index"]

    def ingest_file(
        self,
        source_path: Path,
        *,
        company_id: str | None = None,
        company_name: str | None = None,
        fiscal_year: int | None = None,
    ) -> DocumentRecord:
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        doc_id = new_id()
        source_hash = file_sha256(source_path)
        existing = next(
            (r for r in self.store.load_registry() if r.source_hash == source_hash),
            None,
        )
        if existing:
            logger.info("Skip duplicate file: %s", source_path.name)
            return existing

        norm_dir = self.store.root / "normalized" / doc_id
        pdf_path = normalize_to_pdf(
            source_path,
            norm_dir,
            libreoffice_path=self.pipeline_cfg.get("normalize", {}).get("libreoffice_path"),
        )

        parser = get_parser(self.parse_cfg.get("engine", "auto"))
        figures_dir = self.store.figure_path(doc_id, "_root").parent
        parsed = parser.parse(
            pdf_path,
            parse_version=self.pipeline_cfg.get("parse_version", "v1.0.0"),
            extract_images=self.parse_cfg.get("extract_images", True),
            figures_dir=figures_dir,
            doc_id=doc_id,
        )

        if company_id:
            parsed.company_id = company_id
        if company_name:
            parsed.company_name = company_name
        if fiscal_year:
            parsed.fiscal_year = fiscal_year

        self.store.save_parsed(parsed)

        chunk_builder = ChunkBuilder(
            max_tokens=self.chunk_cfg.get("max_tokens", 512),
            overlap_tokens=self.chunk_cfg.get("overlap_tokens", 64),
            table_max_rows=self.chunk_cfg.get("table_max_rows_per_chunk", 40),
            parse_version=self.pipeline_cfg.get("parse_version", "v1.0.0"),
        )
        chunks = chunk_builder.build(parsed)
        self.store.save_chunks(doc_id, chunks)

        record = DocumentRecord(
            doc_id=doc_id,
            company_id=parsed.company_id or "unknown",
            company_name=parsed.company_name or source_path.stem,
            fiscal_year=parsed.fiscal_year or 0,
            source_filename=source_path.name,
            source_hash=source_hash,
            storage_uri=str(source_path),
            parse_status="parsed",
            parse_version=parsed.parse_version,
            chunk_count=len(chunks),
        )
        self.store.upsert_registry(record)
        logger.info(
            "Ingested %s -> %s chunks (company=%s, year=%s)",
            source_path.name,
            len(chunks),
            record.company_name,
            record.fiscal_year,
        )
        return record

    def ingest_directory(self, directory: Path | None = None) -> list[DocumentRecord]:
        directory = directory or self.raw_dir
        records: list[DocumentRecord] = []
        patterns = ("*.pdf", "*.doc", "*.docx")
        files: list[Path] = []
        for pattern in patterns:
            files.extend(sorted(directory.glob(pattern)))
        for path in files:
            records.append(self.ingest_file(path))
        return records

    def rebuild_indexes(self) -> dict[str, int]:
        chunks = self.store.load_all_chunks()
        vector = VectorIndex(self.store)
        bm25 = BM25Index(self.store)
        v_count = vector.rebuild(
            chunks,
            batch_size=self.index_cfg.get("batch_embedding_size", 32),
        )
        b_count = bm25.rebuild(chunks)
        return {"vector": v_count, "bm25": b_count}

    def run_full(self, directory: Path | None = None) -> dict:
        records = self.ingest_directory(directory)
        index_stats = self.rebuild_indexes()
        return {"documents": [r.model_dump(mode="json") for r in records], "index": index_stats}
