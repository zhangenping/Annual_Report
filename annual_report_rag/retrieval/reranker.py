"""Cross-encoder reranker."""

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
            logger.warning("Rerank failed, fallback to hybrid score: %s", exc)
            return candidates[:top_k]
