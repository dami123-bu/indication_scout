"""Application configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str
    db_password: str

    # API Keys
    openai_api_key: str = ""
    pubmed_api_key: str = ""
    anthropic_api_key: str = ""
    ncbi_api_key: str = ""
    openfda_api_key: str = ""

    # LLM Settings
    llm_model: str = "claude-sonnet-4-6"
    small_llm_model: str = "claude-haiku-4-5-20251001"
    embedding_model: str = "text-embedding-3-small"

    # App Settings
    debug: bool = False
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        frozen = True


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
