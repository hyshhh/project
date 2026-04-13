"""配置管理 — 所有参数收归此处，从环境变量 / .env 文件加载"""

from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """对话模型配置"""
    model: str = Field(default="Qwen/Qwen3-VL-4B-AWQ", alias="CHAT_MODEL")
    api_key: str = Field(default="abc123", alias="LLM_API_KEY")
    base_url: str = Field(default="http://localhost:7890/v1", alias="LLM_BASE_URL")
    temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")

    model_config = {"populate_by_name": True}


class EmbedConfig(BaseSettings):
    """Embedding 模型配置"""
    model: str = Field(default="text-embedding-v4", alias="EMBED_MODEL")
    api_key: str = Field(default="", alias="EMBED_API_KEY")
    base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="EMBED_BASE_URL",
    )

    model_config = {"populate_by_name": True}


class RetrievalConfig(BaseSettings):
    """RAG 检索参数"""
    top_k: int = Field(default=3, alias="RETRIEVAL_TOP_K")
    score_threshold: float = Field(default=0.5, alias="RETRIEVAL_SCORE_THRESHOLD")

    model_config = {"populate_by_name": True}


class VectorStoreConfig(BaseSettings):
    """向量库持久化配置"""
    persist_path: str = Field(default="./vector_store", alias="VECTOR_STORE_PATH")
    auto_rebuild: bool = Field(default=False, alias="VECTOR_STORE_AUTO_REBUILD")

    model_config = {"populate_by_name": True}


class AppConfig(BaseSettings):
    """全局应用配置"""
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    ship_db_path: str | None = Field(default=None, alias="SHIP_DB_PATH")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)

    model_config = {"populate_by_name": True}


def load_config(env_file: str | Path | None = None) -> AppConfig:
    """加载配置。优先从 .env 文件读取。"""
    if env_file is None:
        candidate = Path.cwd() / ".env"
        env_file = candidate if candidate.exists() else None
    if env_file:
        return AppConfig(_env_file=str(env_file))
    return AppConfig()
