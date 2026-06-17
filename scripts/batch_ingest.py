#!/usr/bin/env python
"""
批量入库 CLI。

用法：
  python scripts/batch_ingest.py
  python scripts/batch_ingest.py --input-dir path/to/reports
  python scripts/batch_ingest.py --skip-index   # 仅解析切片，跳过 Embedding

等价于执行 IngestPipeline.run_full() 的交互式版本。
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annual_report_rag.pipelines.ingest.pipeline import IngestPipeline

console = Console()


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="原始年报目录，默认 configs/pipelines.yaml 中 raw_dir",
)
@click.option("--skip-index", is_flag=True, help="仅解析切片，不重建索引")
def main(input_dir: Path | None, skip_index: bool) -> None:
    pipeline = IngestPipeline()
    console.print("[bold green]开始批量入库...[/bold green]")
    records = pipeline.ingest_directory(input_dir)

    table = Table(title="入库结果")
    table.add_column("文件名")
    table.add_column("公司")
    table.add_column("年份")
    table.add_column("Chunks")
    table.add_column("状态")
    for r in records:
        table.add_row(
            r.source_filename,
            r.company_name,
            str(r.fiscal_year or "-"),
            str(r.chunk_count),
            r.parse_status,
        )
    console.print(table)

    if not skip_index:
        console.print("[bold blue]重建向量与 BM25 索引...[/bold blue]")
        stats = pipeline.rebuild_indexes()
        console.print(f"索引完成: {stats}")


if __name__ == "__main__":
    main()
