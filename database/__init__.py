"""船弦号数据库 — FAISS 向量库 + 精确查找，标准 RAG 模式"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from config import EmbedConfig, RetrievalConfig, VectorStoreConfig

logger = logging.getLogger(__name__)


class DashScopeEmbeddings(Embeddings):
    """DashScope Embedding 封装，直接调用 OpenAI 兼容模式 API。"""

    def __init__(self, model: str, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.model = model
        self.api_key = api_key
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import httpx
        resp = httpx.post(
            self._url,
            headers=self._headers,
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        if resp.status_code != 200:
            logger.error("DashScope embedding 请求失败 [%d]: %s", resp.status_code, resp.text)
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


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
    # "0901": "黑色滚装船，船尾有巨大跳板，侧舷有汽车装载门",
    "0921": "黑色",
}


class ShipDatabase:
    """
    船弦号数据库 — 双通道检索：
      1. 精确查找（dict，O(1)）
      2. FAISS 向量语义检索（RAG）

    每条记录 = Document(
        page_content="弦号 {hn}\n{description}",   # 用于 embedding
        metadata={"hull_number": hn, "description": desc}
    )
    """

    def __init__(
        self,
        embed_config: EmbedConfig,
        retrieval_config: RetrievalConfig,
        vector_store_config: VectorStoreConfig,
        data: dict[str, str] | None = None,
        db_path: str | None = None,
    ):
        self._embed_config = embed_config
        self._retrieval_config = retrieval_config
        self._vs_config = vector_store_config

        # ── 加载数据源 ──
        if db_path and Path(db_path).exists():
            self._data = self._load_json(db_path)
            logger.info("从 %s 加载了 %d 条船记录", db_path, len(self._data))
        elif data:
            self._data = dict(data)
        else:
            self._data = dict(DEFAULT_SHIP_DB)
            logger.info("使用内置数据库，共 %d 条记录", len(self._data))

        # ── Embedding 客户端 ──
        self._embeddings = DashScopeEmbeddings(
            model=self._embed_config.model,
            api_key=self._embed_config.api_key,
            base_url=self._embed_config.base_url,
        )

        # ── 向量库（懒加载） ──
        self._vector_store: FAISS | None = None

    # ── 数据加载 ──────────────────────────────

    @staticmethod
    def _load_json(path: str) -> dict[str, str]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"数据库文件格式错误，期望 dict，实际 {type(raw)}")
        return {str(k): str(v) for k, v in raw.items()}

    # ── 向量库构建 ─────────────────────────────

    def _build_documents(self) -> list[Document]:
        """将所有船记录转为 Document 列表"""
        docs = []
        for hn, desc in self._data.items():
            content = f"弦号 {hn}\n{desc}"
            docs.append(Document(
                page_content=content,
                metadata={"hull_number": hn, "description": desc},
            ))
        return docs

    def _load_or_build_vector_store(self) -> FAISS:
        """尝试从磁盘加载缓存，不存在则重新构建"""
        persist_dir = Path(self._vs_config.persist_path)
        index_file = persist_dir / "index.faiss"

        # 尝试加载缓存
        if (
            not self._vs_config.auto_rebuild
            and index_file.exists()
        ):
            try:
                logger.info("从 %s 加载向量库缓存…", persist_dir)
                vs = FAISS.load_local(
                    str(persist_dir),
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("向量库缓存加载成功")
                return vs
            except Exception as e:
                logger.warning("缓存加载失败（%s），将重新构建", e)

        # 构建新索引
        docs = self._build_documents()
        logger.info("正在构建 FAISS 向量库（%d 条文档）…", len(docs))
        vs = FAISS.from_documents(docs, self._embeddings)

        # 持久化
        persist_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(persist_dir))
        logger.info("向量库已持久化到 %s", persist_dir)

        return vs

    @property
    def vector_store(self) -> FAISS:
        if self._vector_store is None:
            self._vector_store = self._load_or_build_vector_store()
        return self._vector_store

    # ── 精确查找 ──────────────────────────────

    def lookup(self, hull_number: str) -> str | None:
        """精确查找，返回描述或 None。"""
        return self._data.get(hull_number.strip())

    # ── 语义检索 ──────────────────────────────

    def semantic_search(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        FAISS 语义检索，返回 [
            {"hull_number": str, "description": str, "score": float},
            ...
        ]
        score 为归一化相似度，越高越匹配。
        """
        k = top_k or self._retrieval_config.top_k
        results_with_score = self.vector_store.similarity_search_with_score(query, k=k)

        results = []
        for doc, distance in results_with_score:
            # FAISS 内积距离 → 相似度（值越小越相似，转为 0~1）
            score = 1.0 / (1.0 + distance)
            results.append({
                "hull_number": doc.metadata["hull_number"],
                "description": doc.metadata["description"],
                "score": round(score, 4),
            })
        return results

    def semantic_search_filtered(self, query: str) -> list[dict]:
        """带阈值过滤的语义检索"""
        results = self.semantic_search(query, top_k=self._retrieval_config.top_k)
        threshold = self._retrieval_config.score_threshold
        return [r for r in results if r["score"] >= threshold]

    # ── 属性 ──────────────────────────────────

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
