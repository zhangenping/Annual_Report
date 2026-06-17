"""Local filesystem storage for documents, chunks, and registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from annual_report_rag.config import resolve_path
from annual_report_rag.schemas.chunk import Chunk
from annual_report_rag.schemas.document import DocumentRecord, ParsedDocument


class LocalStore:
    def __init__(self, data_dir: str = "data") -> None:
        self.root = resolve_path(data_dir)
        self.parsed_dir = self.root / "parsed"
        self.chunks_dir = self.root / "chunks"
        self.figures_dir = self.root / "figures"
        self.index_dir = self.root / "index"
        self.registry_path = self.root / "documents.json"
        for d in (self.parsed_dir, self.chunks_dir, self.figures_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save_parsed(self, parsed: ParsedDocument) -> Path:
        path = self.parsed_dir / f"{parsed.doc_id}.json"
        path.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load_parsed(self, doc_id: str) -> ParsedDocument:
        path = self.parsed_dir / f"{doc_id}.json"
        return ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))

    def save_chunks(self, doc_id: str, chunks: list[Chunk]) -> Path:
        path = self.chunks_dir / f"{doc_id}.json"
        payload = [c.model_dump(mode="json") for c in chunks]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_chunks(self, doc_id: str) -> list[Chunk]:
        path = self.chunks_dir / f"{doc_id}.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Chunk.model_validate(item) for item in data]

    def load_all_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        for path in sorted(self.chunks_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            chunks.extend(Chunk.model_validate(item) for item in data)
        return chunks

    def load_registry(self) -> list[DocumentRecord]:
        if not self.registry_path.exists():
            return []
        data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        return [DocumentRecord.model_validate(item) for item in data]

    def upsert_registry(self, record: DocumentRecord) -> None:
        records = {r.doc_id: r for r in self.load_registry()}
        records[record.doc_id] = record
        payload = [r.model_dump(mode="json") for r in records.values()]
        self.registry_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def figure_path(self, doc_id: str, name: str) -> Path:
        folder = self.figures_dir / doc_id
        folder.mkdir(parents=True, exist_ok=True)
        return folder / name

    def save_json(self, relative: str, data: Any) -> Path:
        path = self.index_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_json(self, relative: str) -> Any:
        path = self.index_dir / relative
        return json.loads(path.read_text(encoding="utf-8"))
