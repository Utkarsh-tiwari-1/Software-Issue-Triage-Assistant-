"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime settings. Environment variable names match field names."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Issue Triage Assistant"
    environment: Literal["development", "test", "production"] = "development"
    llm_provider: Literal["auto", "local", "openai"] = "auto"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = Field(default=30.0, gt=0, le=180)
    embedding_provider: Literal["auto", "local", "openai"] = "auto"
    openai_embedding_model: str = "text-embedding-3-small"
    database_path: Path = Path("./data/issues.sqlite3")
    chroma_path: Path = Path("./data/chroma")
    retrieval_k: int = Field(default=5, ge=1, le=20)
    duplicate_threshold: float = Field(default=0.82, ge=0, le=1)
    api_key: SecretStr | None = None
    allowed_origins: list[str] = Field(default_factory=list)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                return value
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        return value

    @field_validator("openai_api_key", "api_key", mode="before")
    @classmethod
    def treat_blank_secrets_as_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_providers(self) -> "Settings":
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        return self

    @property
    def use_openai_llm(self) -> bool:
        return self.llm_provider == "openai" or (
            self.llm_provider == "auto" and self.openai_api_key is not None
        )

    @property
    def use_openai_embeddings(self) -> bool:
        return self.embedding_provider == "openai" or (
            self.embedding_provider == "auto" and self.openai_api_key is not None
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings for the default application instance."""

    return Settings()
