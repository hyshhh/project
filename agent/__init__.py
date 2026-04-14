"""Agent 核心 — 构建与运行，带 few-shot 示例引导链路"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import AppConfig, load_config
from database import ShipDatabase
from tools import build_tools

logger = logging.getLogger(__name__)

# ── System Prompt ──────────────────────────────

SYSTEM_PROMPT = """你是船弦号识别助手。严格按以下流程工作：

【流程规则】
1. 用户提供了弦号：
   → 第一步：调用 lookup_by_hull_number 精确查找
   → found=true：直接返回「识别结果：弦号 {hull_number}，描述：{description}」
   → found=false：进入第二步
   → 第二步：调用 retrieve_by_description 语义检索，用用户的完整描述作为查询

2. 用户只提供了描述、没有弦号：
   → 直接调用 retrieve_by_description，用用户的描述作为查询

3. 语义检索结果解读：
   → 有结果：返回「根据描述检索到最相似的船：弦号 {hull_number}，描述：{description}（相似度：{score}）」
   → 无结果：返回「未找到匹配的船只信息」

【禁止】
- 不要编造弦号或描述
- 不要跳过精确查找直接语义检索
- 不要同时调用两个工具
"""

# ── Few-shot 示例 ──────────────────────────────

FEW_SHOT_EXAMPLES = [
    # 示例 1：精确匹配
    HumanMessage(content="帮我查一下弦号0014是什么船"),
    AIMessage(content="", tool_calls=[
        {"id": "call_1", "name": "lookup_by_hull_number", "args": {"hull_number": "0014"}}
    ]),
    ToolMessage(
        content='{"found": true, "hull_number": "0014", "description": "白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪"}',
        tool_call_id="call_1",
    ),
    AIMessage(content="识别结果：弦号 0014，描述：白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪"),

    # 示例 2：精确查不到 → 语义检索
    HumanMessage(content="弦号9999，这是一艘大型白色邮轮，船身有蓝色条纹装饰，有三个烟囱"),
    AIMessage(content="", tool_calls=[
        {"id": "call_2", "name": "lookup_by_hull_number", "args": {"hull_number": "9999"}}
    ]),
    ToolMessage(
        content='{"found": false, "hull_number": "9999"}',
        tool_call_id="call_2",
    ),
    AIMessage(content="", tool_calls=[
        {"id": "call_3", "name": "retrieve_by_description", "args": {
            "target_description": "大型白色邮轮，船身有蓝色条纹装饰，有三个烟囱"
        }}
    ]),
    ToolMessage(
        content='{"results": [{"hull_number": "0123", "description": "白色邮轮，船身有红蓝条纹装饰，三座烟囱", "score": 0.9234}]}',
        tool_call_id="call_3",
    ),
    AIMessage(content="未找到对应弦号，根据描述检索到最相似的船：弦号 0123，描述：白色邮轮，船身有红蓝条纹装饰，三座烟囱（相似度：0.9234）"),
]


# ── Agent 类 ────────────────────────────────────

class ShipHullAgent:
    """船弦号识别 Agent 封装。"""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()

        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

        self.db = ShipDatabase(
            embed_config=self.config.embed,
            retrieval_config=self.config.retrieval,
            vector_store_config=self.config.vector_store,
            db_path=self.config.ship_db_path,
        )
        self.tools = build_tools(self.db)

        self._llm = ChatOpenAI(
            model=self.config.llm.model,
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url,
            temperature=self.config.llm.temperature,
        )

        self._agent = create_react_agent(
            model=self._llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
        )

    def run(self, query: str) -> str:
        """执行一次识别查询，返回最终回答文本。"""
        logger.info("收到查询: %s", query)
        messages = FEW_SHOT_EXAMPLES + [HumanMessage(content=query)]
        result = self._agent.invoke({"messages": messages})
        answer = result["messages"][-1].content
        logger.info("回答: %s", answer)
        return answer

    def run_verbose(self, query: str) -> list[dict]:
        """执行查询，返回完整消息链（调试用）。"""
        messages = FEW_SHOT_EXAMPLES + [HumanMessage(content=query)]
        result = self._agent.invoke({"messages": messages})
        trace = []
        for msg in result["messages"]:
            entry = {"type": msg.type, "content": msg.content}
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = [
                    {"name": tc["name"], "args": tc["args"]}
                    for tc in msg.tool_calls
                ]
            if hasattr(msg, "tool_call_id"):
                entry["tool_call_id"] = msg.tool_call_id
            trace.append(entry)
        return trace


# ── 便捷工厂 ────────────────────────────────────

_agent_instance: ShipHullAgent | None = None


def create_agent(config: AppConfig | None = None) -> ShipHullAgent:
    """创建 Agent 实例（单例）。"""
    global _agent_instance
    if _agent_instance is None or config is not None:
        _agent_instance = ShipHullAgent(config)
    return _agent_instance