"""Application settings and config loaders."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default=PROJECT_ROOT)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", alias="OPENAI_BASE_URL"
    )
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_yaml_config(name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "configs" / name
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(relative: str) -> Path:
    return (PROJECT_ROOT / relative).resolve()
