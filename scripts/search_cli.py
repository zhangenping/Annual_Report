#!/usr/bin/env python
"""命令行检索测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annual_report_rag.retrieval import HybridSearch, SearchFilters


@click.command()
@click.argument("query")
@click.option("--top-k", default=5, show_default=True)
@click.option("--company-id", default=None)
@click.option("--year", type=int, default=None)
def main(query: str, top_k: int, company_id: str | None, year: int | None) -> None:
    search = HybridSearch()
    filters = SearchFilters(
        company_id=company_id,
        fiscal_years=[year] if year else [],
    )
    hits = search.search(query, filters=filters, top_k=top_k, rerank=True)
    click.echo(json.dumps(hits, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
