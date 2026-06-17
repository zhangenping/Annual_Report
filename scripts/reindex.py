#!/usr/bin/env python
"""
仅重建向量/BM25 索引（不重新解析 PDF）。

适用场景：修改 Embedding 模型、调整检索权重后，基于已有 data/chunks/ 快速重索引。
"""

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
