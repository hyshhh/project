"""船弦号数据库 — 支持内置数据和外部 JSON 文件加载"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping

import numpy as np
from langchain_openai import OpenAIEmbeddings

from config import EmbedConfig

logger = logging.getLogger(__name__)

# ── 内置默认数据库 ──────────────────────────────

DEFAULT_SHIP_DB: dict[str, str] = {
    "0014": "白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪",
    "0025": "黑色散货船，船体有红色水线，甲板上配有龙门吊",
    "0123": "白色邮轮，船身有红蓝条纹装饰，三座烟囱",
    "0256": "灰色军舰，隐身外形设计，舰首配有垂直发射系统",
    "0389": "红色渔船，船身有白色编号，甲板配有拖网绞车",
    "0455": "绿色集装箱船，船体涂有大型LOGO，配有四台岸桥吊",
    "0512": "黄色挖泥船，船体宽大，中部有大型绞吸臂",
    "0678": "蓝色油轮，双壳结构，船尾有大型舵机舱",
    "0789": "白色科考船，船尾有A型吊架，甲板有多个实验室舱",
    "0901": "黑色滚装船，船尾有巨大跳板，侧舷有汽车装载门",
}


class ShipDatabase:
    """管理弦号 → 描述映射，支持动态加载外部 JSON。"""

    def __init__(self, data: dict[str, str] | None = None, db_path: str | None = None):
        if db_path and Path(db_path).exists():
            self._data = self._load_json(db_path)
            logger.info("从 %s 加载了 %d 条船记录", db_path, len(self._data))
        elif data:
            self._data = dict(data)
        else:
            self._data = dict(DEFAULT_SHIP_DB)
            logger.info("使用内置数据库，共 %d 条记录", len(self._data))

    @staticmethod
    def _load_json(path: str) -> dict[str, str]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"数据库文件格式错误，期望 dict，实际 {type(raw)}")
        return {str(k): str(v) for k, v in raw.items()}

    def lookup(self, hull_number: str) -> str | None:
        """精确查找，返回描述或 None。"""
        return self._data.get(hull_number.strip())

    @property
    def hull_numbers(self) -> list[str]:
        return list(self._data.keys())

    @property
    def descriptions(self) -> list[str]:
        return list(self._data.values())

    @property
    def items(self) -> Mapping[str, str]:
        return self._data

    def __len__(self) -> int:
        return len(self._data)


class EmbeddingIndex:
    """文档向量索引 — 懒加载，只调用一次 embedding API。"""

    def __init__(self, config: EmbedConfig, descriptions: list[str]):
        self._config = config
        self._descriptions = descriptions
        self._client: OpenAIEmbeddings | None = None
        self._matrix: np.ndarray | None = None

    def _ensure_client(self) -> OpenAIEmbeddings:
        if self._client is None:
            self._client = OpenAIEmbeddings(
                model=self._config.model,
                api_key=self._config.api_key,
                base_url=self._config.base_url,
            )
        return self._client

    def _ensure_index(self) -> np.ndarray:
        if self._matrix is None:
            logger.info("正在构建 embedding 索引（%d 条文档）…", len(self._descriptions))
            client = self._ensure_client()
            embs = client.embed_documents(self._descriptions)
            self._matrix = np.array(embs, dtype=np.float32)
            logger.info("索引构建完成，shape=%s", self._matrix.shape)
        return self._matrix

    def search(self, query: str, top_k: int = 1) -> list[tuple[int, float]]:
        """
        语义检索，返回 [(index, score), ...] 按 score 降序。
        """
        client = self._ensure_client()
        query_emb = np.array(client.embed_query(query), dtype=np.float32)
        doc_matrix = self._ensure_index()

        # cosine similarity
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        d_norms = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-10)
        scores = d_norms @ q_norm

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]
