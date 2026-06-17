"""
格式归一化：将多种输入统一为 PDF。

设计原则（见设计文档）：DOC/DOCX 与 PDF 共用下游解析管线，
避免维护两套切片逻辑。依赖 LibreOffice 无头模式做 Office 转换。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def find_libreoffice() -> str | None:
    """在 PATH 或 Windows 默认安装路径中查找 LibreOffice。"""
    candidates = [
        "soffice",
        "libreoffice",
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ]
    for cmd in candidates:
        if shutil.which(cmd) or Path(cmd).exists():
            return cmd
    return None


def normalize_to_pdf(source: Path, output_dir: Path, libreoffice_path: str | None = None) -> Path:
    """
    输入 PDF 则复制到 output_dir；输入 DOC/DOCX 则转换为 PDF。

    返回归一化后的 PDF 路径，供解析器消费。
    """
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        target = output_dir / source.name
        output_dir.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return target

    if suffix not in {".doc", ".docx"}:
        raise ValueError(f"Unsupported file type: {suffix}")

    lo = libreoffice_path or find_libreoffice()
    if not lo:
        raise RuntimeError("LibreOffice not found; install it to convert DOC/DOCX files.")

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        lo,
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(source),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    pdf_path = output_dir / f"{source.stem}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"Conversion failed for {source}")
    return pdf_path
