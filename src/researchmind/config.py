"""Application configuration, loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentBudgetSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    max_subquestions: int = Field(5, ge=1, le=10)
    max_rounds_per_subquestion: int = Field(3, ge=1, le=6)
    max_followups: int = Field(2, ge=0, le=5)
    max_llm_calls: int = Field(40, ge=5, le=400)
    max_seconds: int = Field(900, ge=30)
    top_k: int = Field(8, ge=1, le=30)
    rerank_candidates: int = Field(40, ge=5, le=200)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+psycopg://researchmind:researchmind@localhost:5432/researchmind"

    # LLM
    llm_provider: Literal["ollama", "gemini", "openai_compat", "fake"] = "ollama"
    llm_model: str = "qwen3.5:9b"
    llm_temperature: float = 0.1
    llm_timeout_s: float = 300.0
    llm_max_output_tokens: int = 2048
    ollama_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 16384
    ollama_think: bool = False
    gemini_api_key: SecretStr | None = None
    openai_compat_base_url: str | None = None
    openai_compat_api_key: SecretStr | None = None

    # Embeddings
    embedding_provider: Literal["ollama", "sentence_transformers", "hashing"] = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    # Reranker ("none" disables reranking)
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = "cpu"

    # Scholarly APIs
    contact_email: str | None = None
    semantic_scholar_api_key: SecretStr | None = None
    http_cache_ttl_s: int = 7 * 24 * 3600

    # Storage and logging
    data_dir: Path = Path("./data")
    log_level: str = "INFO"
    log_json: bool = True

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def psycopg_url(self) -> str:
        """Plain libpq URL (used by the LangGraph checkpointer)."""
        return self.database_url.replace("postgresql+psycopg://", "postgresql://", 1)

    def ensure_dirs(self) -> None:
        for d in (self.pdf_dir, self.index_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_budget_settings() -> AgentBudgetSettings:
    return AgentBudgetSettings()
