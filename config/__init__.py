"""配置管理 — 所有参数收归此处，从环境变量 / .env 文件加载"""

from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """对话模型配置"""
    model: str = Field(default="Qwen/Qwen3-VL-4B-AWQ", alias="chat_model")
    api_key: str = Field(default="abc123", alias="llm_api_key")
    base_url: str = Field(default="http://localhost:7890/v1", alias="llm_base_url")
    temperature: float = Field(default=0.0, alias="llm_temperature")

    model_config = {"populate_by_name": True, "extra": "ignore"}


class EmbedConfig(BaseSettings):
    """Embedding 模型配置"""
    model: str = Field(default="text-embedding-v3", alias="embed_model")
    api_key: str = Field(default="sk-f33186cbf0934604b51dfd282acb25cb", alias="embed_api_key")
    base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="embed_base_url",
    )

    model_config = {"populate_by_name": True, "extra": "ignore"}


class RetrievalConfig(BaseSettings):
    """RAG 检索参数"""
    top_k: int = Field(default=3, alias="retrieval_top_k")
    score_threshold: float = Field(default=0.5, alias="retrieval_score_threshold")

    model_config = {"populate_by_name": True, "extra": "ignore"}


class VectorStoreConfig(BaseSettings):
    """向量库持久化配置"""
    persist_path: str = Field(default="./vector_store", alias="vector_store_path")
    auto_rebuild: bool = Field(default=False, alias="vector_store_auto_rebuild")

    model_config = {"populate_by_name": True, "extra": "ignore"}


class AppConfig(BaseSettings):
    """全局应用配置"""
    log_level: str = Field(default="INFO", alias="log_level")
    ship_db_path: str | None = Field(default=None, alias="ship_db_path")

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)

    model_config = {"populate_by_name": True, "extra": "ignore"}


def load_config(env_file: str | Path | None = None) -> AppConfig:
    """加载配置。优先从 .env 文件读取。"""
    if env_file is None:
        candidate = Path.cwd() / ".env"
        env_file = candidate if candidate.exists() else None
    if env_file:
        return AppConfig(_env_file=str(env_file))
    return AppConfig()