"""Application configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = "sqlite:///./indication_scout.db"

    # API Keys
    openai_api_key: str = ""
    pubmed_api_key: str = ""
    openfda_api_key: str = ""

    # LLM Settings
    llm_model: str = "gpt-4"
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
