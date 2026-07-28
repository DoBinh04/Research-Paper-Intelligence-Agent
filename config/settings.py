"""Central configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    cohere_api_key: str = ""
    langchain_api_key: str = ""
    semantic_scholar_api_key: str = ""
    openai_base_url: str = ""

    chroma_path: str = "./data/chroma"
    max_citation_hop: int = 2
    max_nodes: int = 50
    cache_ttl: int = 3600
    llm_model: str = "gpt-4o-mini"
    LLM_CHEAP_MODEL: str = "gpt-4o-mini"
    synthesis_model: str = "gpt-4o"
    max_loop_count: int = 3

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    langsmith_tracing: bool = True
    langsmith_project: str = "research-paper-agent"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    use_bge_m3: bool = False

    enable_hybrid: bool = True
    dense_candidates: int = 20
    bm25_candidates: int = 20
    rrf_k: int = 60
    rrf_dense_weight: float = 1.0
    rrf_bm25_weight: float = 1.0

    @property
    def chroma_dir(self) -> Path:
        path = Path(self.chroma_path)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
