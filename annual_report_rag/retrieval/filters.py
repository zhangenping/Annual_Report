"""Retrieval filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchFilters:
    company_id: str | None = None
    company_name: str | None = None
    fiscal_years: list[int] = field(default_factory=list)
    chunk_types: list[str] = field(default_factory=list)
    section_contains: str | None = None

    def chroma_where(self) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if self.company_id:
            clauses.append({"company_id": self.company_id})
        if self.fiscal_years:
            if len(self.fiscal_years) == 1:
                clauses.append({"fiscal_year": self.fiscal_years[0]})
        if self.chunk_types:
            if len(self.chunk_types) == 1:
                clauses.append({"chunk_type": self.chunk_types[0]})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def match_chunk(self, chunk_meta: dict[str, Any]) -> bool:
        metadata = chunk_meta.get("metadata", chunk_meta)
        if isinstance(metadata, dict) and "company_id" in metadata:
            m = metadata
        else:
            m = chunk_meta

        if self.company_id and m.get("company_id") not in {self.company_id, ""}:
            if m.get("metadata", {}).get("company_id") != self.company_id:
                return False
        if self.fiscal_years:
            year = m.get("fiscal_year") or m.get("metadata", {}).get("fiscal_year")
            if year and int(year) not in self.fiscal_years:
                return False
        if self.chunk_types:
            ctype = m.get("chunk_type") or m.get("metadata", {}).get("chunk_type")
            if ctype and ctype not in self.chunk_types:
                return False
        if self.section_contains:
            section = str(m.get("section", ""))
            if self.section_contains not in section:
                return False
        return True
