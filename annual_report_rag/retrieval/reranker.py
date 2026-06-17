"""
Cross-Encoder 精排（Reranker）。

粗召回（向量+BM25）侧重召回率，Reranker 用 query-document 交叉注意力做精排，
显著提升 Top-K 准确率。模型懒加载并缓存，避免重复初始化。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from annual_report_rag.config import load_yaml_config

logger = logging.getLogger(__name__)


@lru_cache
def _get_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class Reranker:
    """对混合检索候选集做二次排序。"""

    def __init__(self) -> None:
        cfg = load_yaml_config("models.yaml")["reranker"]
        self.enabled = cfg.get("enabled", True)
        self.model_name = cfg.get("model_name", "BAAI/bge-reranker-base")
        self.top_k = cfg.get("top_k_after_rerank", 8)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _get_reranker(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled or not candidates:
            return candidates[: top_k or self.top_k]

        top_k = top_k or self.top_k
        pairs = []
        for c in candidates:
            text = c.get("text") or c.get("content_text") or ""
            meta = c.get("metadata", {})
            if isinstance(meta, dict) and not text:
                text = meta.get("content_text", "")
            pairs.append((query, text))

        try:
            scores = self.model.predict(pairs)
            ranked = sorted(
                zip(candidates, scores),
                key=lambda x: float(x[1]),
                reverse=True,
            )
            result = []
            for item, score in ranked[:top_k]:
                enriched = dict(item)
                enriched["rerank_score"] = float(score)
                result.append(enriched)
            return result
        except Exception as exc:
            # 模型加载失败时降级为混合分排序，保证服务可用
            logger.warning("Rerank failed, fallback to hybrid score: %s", exc)
            return candidates[:top_k]
