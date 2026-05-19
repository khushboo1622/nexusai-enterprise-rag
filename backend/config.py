from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ────────────────────────────────────────────────
    APP_ENV: str = "development"
    APP_PORT: int = 5050

    # ── Auth ───────────────────────────────────────────────
    # JWT_SECRET_KEY must be at least 32 chars — used to sign tokens
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ── MongoDB Atlas ──────────────────────────────────────
    MONGO_URI: str
    MONGO_DB_NAME: str = "nexusai"

    # ── Qdrant Cloud ───────────────────────────────────────
    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION: str = "nexusai_docs"

    # ── Embedding model (runs locally, no API key needed) ──
    # BGE-small produces 384-dim vectors, free, no API call
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # ── LLM — Groq (primary) ───────────────────────────────
    MODEL_PROVIDER: str = "groq"
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama3-8b-8192"

    # ── LLM — Gemini (fallback / future) ──────────────────
    GEMINI_API_KEY: str = ""

    # ── LangSmith Observability ────────────────────────────
    # Set LANGCHAIN_TRACING_V2=true to enable tracing
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "nexusai"

    class Config:
        env_file = ".env"
        extra = "ignore"


# @lru_cache means Settings() is created once and reused everywhere
# No re-reading .env on every API call — important for performance
@lru_cache()
def get_settings() -> Settings:
    return Settings()