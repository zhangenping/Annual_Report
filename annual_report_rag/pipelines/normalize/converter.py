"""DOC/DOCX to PDF normalization."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def find_libreoffice() -> str | None:
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
