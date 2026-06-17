#!/usr/bin/env python
"""简单检索评估：输出 Recall@K 代理指标。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annual_report_rag.retrieval import HybridSearch


def main() -> None:
    dataset_path = ROOT / "eval" / "datasets" / "qa_v0.json"
    queries = json.loads(dataset_path.read_text(encoding="utf-8"))
    search = HybridSearch()
    k = 5
    hit = 0
    for item in queries:
        results = search.search(item["query"], top_k=k, rerank=True)
        if results:
            hit += 1
            print(f"[HIT] {item['id']} {item['query']} -> {results[0]['citation']}")
        else:
            print(f"[MISS] {item['id']} {item['query']}")
    print(f"Recall@{k} (proxy): {hit}/{len(queries)}")


if __name__ == "__main__":
    main()
