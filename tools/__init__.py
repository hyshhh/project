"""LangChain 工具定义 — 基于 FAISS 向量库的双通道检索"""

from __future__ import annotations

import json
import logging
from typing import Annotated

from langchain_core.tools import tool

from database import ShipDatabase

logger = logging.getLogger(__name__)


def build_tools(db: ShipDatabase) -> list:
    """构建带绑定数据库的工具列表。"""

    @tool
    def lookup_by_hull_number(
        hull_number: Annotated[str, "要查询的船弦号，例如 '0014'"],
    ) -> str:
        """
        通过弦号精确查找船只描述。
        优先调用此工具。返回 found=true 则直接回答；found=false 再调用 retrieve_by_description。
        """
        hull_number = hull_number.strip()
        desc = db.lookup(hull_number)
        if desc is not None:
            return json.dumps(
                {"found": True, "hull_number": hull_number, "description": desc},
                ensure_ascii=False,
            )
        return json.dumps({"found": False, "hull_number": hull_number}, ensure_ascii=False)

    @tool
    def retrieve_by_description(
        target_description: Annotated[str, "对目标船的外观文字描述，越详细越好"],
    ) -> str:
        """
        基于 FAISS 向量库的语义检索。当弦号查不到，或用户只提供了描述时调用。
        输入对船只的外观描述，返回 Top-K 最匹配的结果（含相似度分数）。
        """
        try:
            results = db.semantic_search_filtered(target_description)
            if not results:
                # 阈值过滤无结果时，返回 Top-1 原始结果作为参考
                raw = db.semantic_search(target_description, top_k=1)
                if raw:
                    return json.dumps(
                        {"note": "以下结果相似度较低，仅供参考", "results": raw},
                        ensure_ascii=False,
                    )
                return json.dumps({"error": "未找到匹配结果"}, ensure_ascii=False)

            return json.dumps({"results": results}, ensure_ascii=False)
        except Exception as e:
            logger.exception("语义检索失败")
            return json.dumps({"error": f"语义检索失败: {e}"}, ensure_ascii=False)

    return [lookup_by_hull_number, retrieve_by_description]
