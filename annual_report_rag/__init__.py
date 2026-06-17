"""
企业级年报 RAG 知识库。

模块概览：
  pipelines/   入库：归一化 → 解析 → 切片 → 索引
  retrieval/   混合检索 + Rerank
  agents/      LangGraph 问答编排
  api/         FastAPI 接口
  schemas/     数据模型
  storage/     本地持久化（PoC）
"""

__version__ = "0.1.0"
