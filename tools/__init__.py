"""LangChain 工具定义"""

from __future__ import annotations

import json
import logging
from typing import Annotated

from langchain_core.tools import tool

from database import ShipDatabase, EmbeddingIndex

logger = logging.getLogger(__name__)


def build_tools(db: ShipDatabase, index: EmbeddingIndex) -> list:
    """构建带绑定数据库/索引的工具列表。"""

    @tool
    def lookup_by_hull_number(
        hull_number: Annotated[str, "要查询的船弦号，例如 '0014'"],
    ) -> str:
        """通过弦号精确查找船只描述。先调用本工具查弦号，查不到再用语义检索。"""
        hull_number = hull_number.strip()
        desc = db.lookup(hull_number)
        if desc is not None:
            return json.dumps(
                {"found": True, "hull_number": hull_number, "description": desc},
                ensure_ascii=False,
            )
        return json.dumps({"found": False, "hull_number": hull_number}, ensure_ascii=False)

    @tool
    def search_by_description(
        target_description: Annotated[str, "对目标船的外观文字描述"],
    ) -> str:
        """
        根据目标船的外观描述，用语义相似度检索数据库中最匹配的船。
        仅在弦号查不到时使用，或用户只提供了描述没有提供弦号时使用。
        返回 Top-1 最相似的结果。
        """
        try:
            results = index.search(target_description, top_k=1)
            if not results:
                return json.dumps({"error": "未找到匹配结果"}, ensure_ascii=False)

            idx, score = results[0]
            hn = db.hull_numbers[idx]
            return json.dumps(
                [
                    {
                        "hull_number": hn,
                        "description": db.descriptions[idx],
                        "score": round(score, 4),
                    }
                ],
                ensure_ascii=False,
            )
        except Exception as e:
            logger.exception("语义检索失败")
            return json.dumps({"error": f"语义检索失败: {e}"}, ensure_ascii=False)

    return [lookup_by_hull_number, search_by_description]
