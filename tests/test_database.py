"""数据库和工具的单元测试（不需要 LLM / Embedding API）"""

import json
import pytest

from database import ShipDatabase, DEFAULT_SHIP_DB


class TestShipDatabase:
    """测试 ShipDatabase 的精确查找逻辑。"""

    def test_lookup_existing(self):
        db = ShipDatabase()
        assert db.lookup("0014") == DEFAULT_SHIP_DB["0014"]

    def test_lookup_missing(self):
        db = ShipDatabase()
        assert db.lookup("9999") is None

    def test_lookup_whitespace(self):
        db = ShipDatabase()
        assert db.lookup("  0014  ") == DEFAULT_SHIP_DB["0014"]

    def test_len(self):
        db = ShipDatabase()
        assert len(db) == len(DEFAULT_SHIP_DB)

    def test_hull_numbers(self):
        db = ShipDatabase()
        assert set(db.hull_numbers) == set(DEFAULT_SHIP_DB.keys())

    def test_custom_data(self):
        custom = {"A001": "测试船"}
        db = ShipDatabase(data=custom)
        assert db.lookup("A001") == "测试船"
        assert len(db) == 1

    def test_load_from_json(self, tmp_path):
        data = {"X001": "JSON船", "X002": "另一艘"}
        p = tmp_path / "ships.json"
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        db = ShipDatabase(db_path=str(p))
        assert db.lookup("X001") == "JSON船"
        assert len(db) == 2

    def test_load_bad_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="格式错误"):
            ShipDatabase(db_path=str(p))
