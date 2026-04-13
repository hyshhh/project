"""数据库和配置的单元测试（不需要 LLM / Embedding API）"""

import json
import pytest

from config import LLMConfig, EmbedConfig, RetrievalConfig, VectorStoreConfig, AppConfig
from database import ShipDatabase, DEFAULT_SHIP_DB


# ══════════════════════════════════════════════
#  Config 测试
# ══════════════════════════════════════════════

class TestLLMConfig:
    def test_defaults(self):
        c = LLMConfig()
        assert c.model == "Qwen/Qwen3-VL-4B-AWQ"
        assert c.api_key == "abc123"
        assert c.temperature == 0.0

    def test_override(self):
        c = LLMConfig(CHAT_MODEL="gpt-4o", LLM_TEMPERATURE=0.7)
        assert c.model == "gpt-4o"
        assert c.temperature == 0.7


class TestEmbedConfig:
    def test_defaults(self):
        c = EmbedConfig()
        assert c.model == "text-embedding-v4"
        assert "dashscope" in c.base_url


class TestRetrievalConfig:
    def test_defaults(self):
        c = RetrievalConfig()
        assert c.top_k == 3
        assert c.score_threshold == 0.5

    def test_override(self):
        c = RetrievalConfig(RETRIEVAL_TOP_K=5, RETRIEVAL_SCORE_THRESHOLD=0.7)
        assert c.top_k == 5
        assert c.score_threshold == 0.7


class TestVectorStoreConfig:
    def test_defaults(self):
        c = VectorStoreConfig()
        assert c.persist_path == "./vector_store"
        assert c.auto_rebuild is False


class TestAppConfig:
    def test_defaults(self):
        c = AppConfig()
        assert c.log_level == "INFO"
        assert c.ship_db_path is None
        assert isinstance(c.llm, LLMConfig)
        assert isinstance(c.embed, EmbedConfig)
        assert isinstance(c.retrieval, RetrievalConfig)
        assert isinstance(c.vector_store, VectorStoreConfig)


# ══════════════════════════════════════════════
#  ShipDatabase 精确查找测试
# ══════════════════════════════════════════════

def _make_db(**kwargs) -> ShipDatabase:
    """创建测试用 ShipDatabase（跳过向量库构建）"""
    return ShipDatabase(
        embed_config=EmbedConfig(api_key="test"),
        retrieval_config=RetrievalConfig(),
        vector_store_config=VectorStoreConfig(persist_path="/tmp/test_vs"),
        **kwargs,
    )


class TestShipDatabase:
    def test_lookup_existing(self):
        db = _make_db()
        assert db.lookup("0014") == DEFAULT_SHIP_DB["0014"]

    def test_lookup_missing(self):
        db = _make_db()
        assert db.lookup("9999") is None

    def test_lookup_whitespace(self):
        db = _make_db()
        assert db.lookup("  0014  ") == DEFAULT_SHIP_DB["0014"]

    def test_len(self):
        db = _make_db()
        assert len(db) == len(DEFAULT_SHIP_DB)

    def test_hull_numbers(self):
        db = _make_db()
        assert set(db.hull_numbers) == set(DEFAULT_SHIP_DB.keys())

    def test_custom_data(self):
        custom = {"A001": "测试船"}
        db = _make_db(data=custom)
        assert db.lookup("A001") == "测试船"
        assert len(db) == 1

    def test_load_from_json(self, tmp_path):
        data = {"X001": "JSON船", "X002": "另一艘"}
        p = tmp_path / "ships.json"
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        db = _make_db(db_path=str(p))
        assert db.lookup("X001") == "JSON船"
        assert len(db) == 2

    def test_load_bad_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="格式错误"):
            _make_db(db_path=str(p))

    def test_build_documents(self):
        db = _make_db()
        docs = db._build_documents()
        assert len(docs) == len(DEFAULT_SHIP_DB)
        for doc in docs:
            assert "弦号" in doc.page_content
            assert "hull_number" in doc.metadata
            assert "description" in doc.metadata
