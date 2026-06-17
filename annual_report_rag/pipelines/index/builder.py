"""
索引构建：向量库（ChromaDB）+ 关键词库（BM25）。

向量索引：
  - Embedding 文本使用 Chunk.embedding_text()，含章节路径前缀（Contextual Retrieval）
  - Chroma payload 存过滤字段（company_id、fiscal_year、chunk_type）

BM25 索引：
  - jieba 分词 + rank_bm25，弥补向量检索对精确财经术语/数字的不足
  - 元数据序列化到 data/index/bm25_registry.json，启动时无需重算分词
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from annual_report_rag.config import load_yaml_config
from annual_report_rag.schemas.chunk import Chunk
from annual_report_rag.storage import LocalStore

logger = logging.getLogger(__name__)


class EmbeddingService:
    """封装 sentence-transformers，模型名从 configs/models.yaml 读取。"""

    def __init__(self) -> None:
        cfg = load_yaml_config("models.yaml")["embedding"]
        from sentence_transformers import SentenceTransformer

        self.model_name = cfg["model_name"]
        self.normalize = cfg.get("normalize", True)
        device = cfg.get("device", "cpu")
        self.model = SentenceTransformer(self.model_name, device=device)

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 20,
        )
        return vectors.tolist()


class VectorIndex:
    """ChromaDB 持久化向量索引，余弦相似度空间。"""

    def __init__(self, store: LocalStore) -> None:
        self.store = store
        chroma_path = str(store.index_dir / "chroma")
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="annual_report_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = EmbeddingService()

    def rebuild(self, chunks: list[Chunk], batch_size: int = 32) -> int:
        """全量重建：先清空 collection 再批量写入。"""
        existing = self.collection.get()
        if existing and existing.get("ids"):
            self.collection.delete(ids=existing["ids"])

        if not chunks:
            return 0

        # embedding_text 含章节前缀，提升语义检索准确率
        texts = [c.embedding_text() for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [_chunk_payload(c) for c in chunks]

        for i in range(0, len(chunks), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            vectors = self.embedder.encode(batch_texts, batch_size=batch_size)
            self.collection.add(
                ids=batch_ids,
                embeddings=vectors,
                documents=batch_texts,
                metadatas=batch_meta,
            )

        logger.info("Indexed %s chunks into ChromaDB", len(chunks))
        return len(chunks)

    def query(
        self,
        query_text: str,
        top_k: int = 50,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """向量检索；where 为 Chroma 元数据过滤（公司/年份等）。"""
        vector = self.embedder.encode([query_text])[0]
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        if not result["ids"] or not result["ids"][0]:
            return hits
        for idx, chunk_id in enumerate(result["ids"][0]):
            distance = result["distances"][0][idx]
            score = 1 - distance  # 余弦距离转相似度
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "score": score,
                    "text": result["documents"][0][idx],
                    "metadata": result["metadatas"][0][idx],
                }
            )
        return hits


class BM25Index:
    """
    内存 BM25 索引。

    PoC 阶段用 JSON 持久化分词语料映射；生产可换 Elasticsearch/OpenSearch。
    """

    def __init__(self, store: LocalStore) -> None:
        self.store = store
        self.chunk_ids: list[str] = []
        self.corpus: list[list[str]] = []
        self.bm25 = None
        self._meta_by_id: dict[str, dict[str, Any]] = {}

    def rebuild(self, chunks: list[Chunk]) -> int:
        import jieba
        from rank_bm25 import BM25Okapi

        self.chunk_ids = [c.chunk_id for c in chunks]
        self.corpus = [list(jieba.cut(c.embedding_text())) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)
        self._meta_by_id = {c.chunk_id: c.model_dump(mode="json") for c in chunks}
        self.store.save_json(
            "bm25_registry.json",
            {"chunk_ids": self.chunk_ids, "meta": self._meta_by_id},
        )
        return len(chunks)

    def load(self, chunks: list[Chunk] | None = None) -> None:
        """优先从磁盘恢复；传入 chunks 时强制重建。"""
        import jieba
        from rank_bm25 import BM25Okapi

        registry_path = self.store.index_dir / "bm25_registry.json"
        if chunks is None and registry_path.exists():
            data = self.store.load_json("bm25_registry.json")
            self.chunk_ids = data["chunk_ids"]
            self._meta_by_id = data["meta"]
            texts = [
                self._meta_by_id[cid]["content_text"] for cid in self.chunk_ids
            ]
            self.corpus = [list(jieba.cut(t)) for t in texts]
            self.bm25 = BM25Okapi(self.corpus)
            return
        if chunks:
            self.rebuild(chunks)

    def query(self, query_text: str, top_k: int = 50) -> list[dict[str, Any]]:
        import jieba

        if not self.bm25:
            return []
        tokens = list(jieba.cut(query_text))
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.chunk_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        hits = []
        for chunk_id, score in ranked:
            if score <= 0:
                continue
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "score": float(score),
                    "metadata": self._meta_by_id.get(chunk_id, {}),
                }
            )
        return hits


def _chunk_payload(chunk: Chunk) -> dict[str, Any]:
    """Chroma 元数据字段（仅支持 str/int/float，用于 where 过滤）。"""
    m = chunk.metadata
    return {
        "doc_id": chunk.doc_id,
        "chunk_type": chunk.chunk_type.value,
        "company_id": m.company_id or "",
        "company_name": m.company_name or "",
        "fiscal_year": m.fiscal_year or 0,
        "section": m.section or "",
        "page_start": m.page_start or 0,
        "source_file": m.source_file,
    }
