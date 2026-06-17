"""
通用工具函数：哈希、文本清洗、表格转换、切片辅助。

被解析管线、切片器、入库流程共同引用。
"""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path


def file_sha256(path: Path) -> str:
    """流式计算文件 SHA256，用于去重与版本溯源。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def new_id() -> str:
    return str(uuid.uuid4())


def estimate_tokens(text: str) -> int:
    """中文场景下粗略估算 token 数（1 token ≈ 2 字符）。"""
    return max(1, len(text) // 2)


def clean_text(text: str) -> str:
    """合并空白字符，减少 PDF 提取产生的多余换行/空格。"""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def guess_company_from_text(text: str) -> tuple[str | None, str | None, int | None]:
    """
    从正文前 8000 字符正则猜测公司名、股票代码、报告年份。

    准确率有限，入库时建议通过 ingest_file(company_id=...) 手工覆盖。
    """
    company_name = None
    company_id = None
    fiscal_year = None

    name_match = re.search(
        r"([\u4e00-\u9fff]{2,20}(?:股份)?有限公司)\s*(?:\d{4}\s*年)?(?:年度)?报告",
        text[:8000],
    )
    if name_match:
        company_name = name_match.group(1)

    code_match = re.search(r"(?:股票代码|证券代码|代码)[:：\s]*([036]\d{5})", text[:8000])
    if code_match:
        company_id = code_match.group(1)

    year_match = re.search(r"(20\d{2})\s*年(?:度)?\s*报告", text[:8000])
    if year_match:
        fiscal_year = int(year_match.group(1))

    return company_name, company_id, fiscal_year


def table_to_markdown(headers: list[list[str]], rows: list[list[str]]) -> str:
    """将表格行列转为 Markdown，供检索展示与 LLM 阅读。"""
    if not headers and not rows:
        return ""
    flat_headers = [str(h or "") for h in (headers[-1] if headers else [])]
    if not flat_headers and rows:
        flat_headers = [f"列{i+1}" for i in range(len(rows[0]))]

    lines = [
        "| " + " | ".join(flat_headers) + " |",
        "| " + " | ".join(["---"] * len(flat_headers)) + " |",
    ]
    for row in rows:
        padded = [str(c or "") for c in row] + [""] * len(flat_headers)
        lines.append("| " + " | ".join(padded[: len(flat_headers)]) + " |")
    return "\n".join(lines)


def token_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    固定窗口文本切分（字符近似 token）。

    overlap 保证跨块边界的句子在至少一个块中完整出现。
    """
    if not text:
        return []
    max_chars = max_tokens * 2
    overlap_chars = overlap_tokens * 2
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return chunks
