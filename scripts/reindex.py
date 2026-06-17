#!/usr/bin/env python
"""仅重建索引（不重新解析）。"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annual_report_rag.pipelines.ingest.pipeline import IngestPipeline

console = Console()


@click.command()
def main() -> None:
    pipeline = IngestPipeline()
    stats = pipeline.rebuild_indexes()
    console.print(f"[green]Reindex done:[/green] {stats}")


if __name__ == "__main__":
    main()
