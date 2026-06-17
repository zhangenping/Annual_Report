"""
混合检索（Hybrid Search）。

召回公式（configs/models.yaml 可调）：
  final_score = α * vector_score + β * normalized_bm25_score

流程：
  1. 向量 Top-K + BM25 Top-K 粗召回
  2. 按 chunk_id 合并去重，加权求和
  3. SearchFilters 做元数据二次过滤（Chroma where 不支持的复杂条件）
  4. Cross-Encoder Reranker 精排，输出 Top-N

设计动机：年报问答既需要语义理解（「利润下滑原因」），也需要精确词匹配（「600383」「研发费用」）。
"""

from __future__ import annotations

from typing import Any

from annual_report_rag.config import load_yaml_config
from annual_report_rag.pipelines.index.builder import BM25Index, VectorIndex
from annual_report_rag.retrieval.filters import SearchFilters
from annual_report_rag.retrieval.reranker import Reranker
from annual_report_rag.schemas.chunk import Chunk
from annual_report_rag.storage import LocalStore


class HybridSearch:
    """向量 + BM25 混合检索入口，供 API 与 Agent 工具调用。"""

    def __init__(self, store: LocalStore | None = None) -> None:
        self.cfg = load_yaml_config("models.yaml")["retrieval"]
        pipeline_cfg = load_yaml_config("pipelines.yaml")
        self.store = store or LocalStore(pipeline_cfg["paths"]["data_dir"])
        self.vector = VectorIndex(self.store)
        self.bm25 = BM25Index(self.store)
        self.bm25.load()
        self.reranker = Reranker()
        # 内存映射：检索命中后快速取完整 Chunk（含 table_json）
        self._chunk_map = {c.chunk_id: c for c in self.store.load_all_chunks()}

    def refresh(self) -> None:
        """入库或重建索引后刷新内存缓存。"""
        self._chunk_map = {c.chunk_id: c for c in self.store.load_all_chunks()}
        self.bm25.load(list(self._chunk_map.values()))

    def search(
        self,
        query: str,
        *,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        filters = filters or SearchFilters()
        top_k = top_k or self.cfg.get("final_top_k", 10)
        alpha = self.cfg.get("alpha_vector", 0.6)
        beta = self.cfg.get("beta_bm25", 0.4)

        vector_hits = self.vector.query(
            query,
            top_k=self.cfg.get("vector_top_k", 50),
            where=filters.chroma_where(),
        )
        bm25_hits = self.bm25.query(query, top_k=self.cfg.get("bm25_top_k", 50))

        merged: dict[str, dict[str, Any]] = {}

        def add_hit(hit: dict[str, Any], weight: float, source: str) -> None:
            chunk_id = hit["chunk_id"]
            chunk = self._chunk_map.get(chunk_id)
            # Chroma where 只能做简单过滤，复杂条件在此二次校验
            if chunk and not filters.match_chunk(chunk.model_dump(mode="json")):
                return
            if chunk_id not in merged:
                merged[chunk_id] = {
                    "chunk_id": chunk_id,
                    "score": 0.0,
                    "sources": [],
                    "chunk": chunk,
                }
            merged[chunk_id]["score"] += hit["score"] * weight
            merged[chunk_id]["sources"].append(source)

        for hit in vector_hits:
            add_hit(hit, alpha, "vector")
        # BM25 分数归一化到 [0,1]，与向量分数量纲对齐
        max_bm25 = max((h["score"] for h in bm25_hits), default=1.0) or 1.0
        for hit in bm25_hits:
            hit = dict(hit)
            hit["score"] = hit["score"] / max_bm25
            add_hit(hit, beta, "bm25")

        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        candidates: list[dict[str, Any]] = []
        for item in ranked:
            chunk: Chunk | None = item.get("chunk")
            if not chunk:
                continue
            candidates.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "score": item["score"],
                    "sources": item["sources"],
                    "chunk_type": chunk.chunk_type.value,
                    "content_text": chunk.content_text,
                    "content_preview": chunk.content_text[:300],
                    "metadata": chunk.metadata.model_dump(mode="json"),
                    "citation": chunk.citation(),
                    "text": chunk.embedding_text(),
                }
            )

        if rerank and candidates:
            candidates = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        return candidates

    def get_chunk(self, chunk_id: str, include_parent: bool = True) -> dict[str, Any] | None:
        """
        按 ID 取切片详情。

        include_parent=True 时附带 Parent 块，用于 Agent 生成阶段补全上下文。
        """
        chunk = self._chunk_map.get(chunk_id)
        if not chunk:
            return None
        result: dict[str, Any] = {"chunk": chunk.model_dump(mode="json")}
        if include_parent and chunk.parent_chunk_id:
            parent = self._chunk_map.get(chunk.parent_chunk_id)
            if parent:
                result["parent"] = parent.model_dump(mode="json")
        return result
