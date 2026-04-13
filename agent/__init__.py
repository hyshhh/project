"""Agent 核心 — 构建与运行"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import AppConfig, load_config
from database import ShipDatabase, EmbeddingIndex
from tools import build_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是船弦号识别助手。工作流程：

1. 用户提供了弦号 → 先调用 `lookup_by_hull_number` 精确查找。
   - 找到（found=true）→ 直接返回弦号和描述。
   - 未找到（found=false）→ 调用 `search_by_description` 语义检索，返回最可能的一个目标船。
2. 用户只提供了描述、没有提供弦号 → 直接调用 `search_by_description`，返回数据库中最可能的船只。

回答格式：
- 精确匹配成功：「识别结果：弦号 {hull_number}，描述：{description}」
- 语义检索结果：「未找到对应弦号，根据描述检索到最相似的船：弦号 {hull_number}，描述：{description}（相似度：{score}）」
- 都没找到：「未找到匹配的船只信息」
"""


class ShipHullAgent:
    """船弦号识别 Agent 的封装。"""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()

        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

        self.db = ShipDatabase(db_path=self.config.ship_db_path)
        self.index = EmbeddingIndex(self.config.embed, self.db.descriptions)
        self.tools = build_tools(self.db, self.index)

        self._llm = ChatOpenAI(
            model=self.config.llm.model,
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url,
        )

        self._agent = create_react_agent(
            model=self._llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
        )

    def run(self, query: str) -> str:
        """执行一次识别查询，返回最终回答文本。"""
        logger.info("收到查询: %s", query)
        result = self._agent.invoke({"messages": [("user", query)]})
        answer = result["messages"][-1].content
        logger.info("回答: %s", answer)
        return answer


# ── 便捷工厂函数 ──────────────────────────────

_agent_instance: ShipHullAgent | None = None


def create_agent(config: AppConfig | None = None) -> ShipHullAgent:
    """创建 Agent 实例（单例）。"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ShipHullAgent(config)
    return _agent_instance
