# 🚢 Ship Hull Agent — 船弦号识别 Agent

基于 **LangChain + LangGraph + FAISS 向量库** 的智能船弦号识别系统，支持 **精确弦号匹配** 和 **RAG 语义检索** 两种识别模式。

## ✨ 功能特性

- **精确匹配**：输入弦号直接查字典，O(1) 响应
- **RAG 语义检索**：基于 FAISS 向量库，每条船记录（弦号+描述）作为一个 Document 建立索引
- **向量库持久化**：首次构建后缓存到磁盘，后续启动直接加载
- **Few-shot 引导**：内置示例对话，确保 Agent 严格遵循检索流程
- **CLI 工具**：命令行单次查询 / 交互式 REPL / 详细调用链模式
- **全参数配置化**：模型、检索、向量库所有参数通过 `.env` 配置

## 🏗️ 项目结构

```
ship-hull-agent/
├── config/
│   └── __init__.py          # pydantic-settings：LLM / Embedding / 检索 / 向量库 全部参数
├── database/
│   └── __init__.py          # ShipDatabase：精确查找 + FAISS 向量语义检索
├── tools/
│   └── __init__.py          # LangChain @tool：lookup_by_hull_number / retrieve_by_description
├── agent/
│   └── __init__.py          # ShipHullAgent：ReAct Agent + Few-shot 示例
├── cli/
│   ├── __init__.py          # Rich CLI：单次查询 / 交互 REPL / --verbose 调用链
│   └── main.py              # python -m cli.main 入口
├── tests/
│   ├── __init__.py
│   └── test_database.py     # 20 个单元测试（配置 + 数据库）
├── .env.example             # 环境变量模板
├── .gitignore
├── pyproject.toml           # 项目元数据 + 依赖声明
└── README.md
```

## 🔄 Agent 工作流程

```
用户输入
  │
  ├─ 包含弦号？
  │    → lookup_by_hull_number 精确查找
  │         ├─ found=true  → 直接返回结果
  │         └─ found=false → retrieve_by_description 语义检索
  │
  └─ 只有描述？
       → retrieve_by_description 语义检索
            ├─ FAISS 向量相似度匹配（top_k + 阈值过滤）
            └─ 返回最匹配的弦号 + 描述 + 相似度
```

### RAG 向量库设计

每条船记录构建为一个 Document：

```
page_content = "弦号 0014\n白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪"
metadata     = {"hull_number": "0014", "description": "白色大型客轮..."}
```

- `page_content` 用于 Embedding 向量化（弦号+描述一起编码，语义更丰富）
- `metadata` 用于提取结构化结果
- FAISS 索引构建后持久化到 `VECTOR_STORE_PATH`，下次启动直接加载

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/hyshhh/project.git
cd project
```

### 2. 安装依赖

```bash
pip install -e .
# 开发模式（含测试）
pip install -e ".[dev]"
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```env
# 对话模型（兼容 OpenAI API 格式）
CHAT_MODEL=Qwen/Qwen3-VL-4B-AWQ
LLM_API_KEY=your-llm-api-key
LLM_BASE_URL=http://localhost:7890/v1

# Embedding 模型
EMBED_MODEL=text-embedding-v4
EMBED_API_KEY=your-embed-api-key
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# RAG 检索参数
RETRIEVAL_TOP_K=3
RETRIEVAL_SCORE_THRESHOLD=0.5

# 向量库持久化路径
VECTOR_STORE_PATH=./vector_store
```

### 4. 运行

```bash
# 单次查询
ship-hull "帮我查一下弦号0014是什么船"

# 交互模式
ship-hull --interactive

# 详细模式（显示工具调用链）
ship-hull --verbose "弦号0256"
```

## 📖 使用示例

### 精确匹配

```
$ ship-hull "帮我查一下弦号0014是什么船"

识别结果：弦号 0014，描述：白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪
```

### 语义检索（弦号不存在）

```
$ ship-hull "弦号9999，这是一艘大型白色邮轮，船身有蓝色条纹装饰，有三个烟囱"

未找到对应弦号，根据描述检索到最相似的船：弦号 0123，描述：白色邮轮，船身有红蓝条纹装饰，三座烟囱（相似度：0.9234）
```

### 详细模式（调试）

```
$ ship-hull --verbose "我看到一艘灰色的军舰，外形很隐身"

┌─────────────────── 🔧 Agent 调用链 ───────────────────┐
│ # │ 类型     │ 内容                                   │
├───┼──────────┼────────────────────────────────────────┤
│ 0 │ human    │ 我看到一艘灰色的军舰，外形很隐身         │
│ 1 │ ai       │ → lookup_by_hull_number({"hull_number… │
│ 2 │ tool     │ ← {"found": false, "hull_number": ""}  │
│ 3 │ ai       │ → retrieve_by_description({"target_de… │
│ 4 │ tool     │ ← {"results": [{"hull_number": "0256"… │
│ 5 │ ai       │ 未找到对应弦号，根据描述检索到最相似的船… │
└─────────────────────────────────────────────────────────┘
```

### 作为 Python 库调用

```python
from agent import create_agent

agent = create_agent()
answer = agent.run("弦号0014是什么船")
print(answer)
```

## ⚙️ 配置说明

### 环境变量一览

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CHAT_MODEL` | 对话模型名称 | `Qwen/Qwen3-VL-4B-AWQ` |
| `LLM_API_KEY` | 对话模型 API Key | `abc123` |
| `LLM_BASE_URL` | 对话模型地址 | `http://localhost:7890/v1` |
| `LLM_TEMPERATURE` | 生成温度（0=确定性最高） | `0.0` |
| `EMBED_MODEL` | Embedding 模型名称 | `text-embedding-v4` |
| `EMBED_API_KEY` | Embedding API Key | — |
| `EMBED_BASE_URL` | Embedding 服务地址 | DashScope |
| `RETRIEVAL_TOP_K` | 语义检索返回条数 | `3` |
| `RETRIEVAL_SCORE_THRESHOLD` | 相似度阈值（低于此值的结果被过滤） | `0.5` |
| `VECTOR_STORE_PATH` | FAISS 索引持久化路径 | `./vector_store` |
| `VECTOR_STORE_AUTO_REBUILD` | 每次启动重建索引 | `false` |
| `SHIP_DB_PATH` | 自定义数据库 JSON 路径 | 内置默认数据 |
| `LOG_LEVEL` | 日志级别 | `INFO` |

### 自定义数据库

创建 JSON 文件（如 `data/ships.json`）：

```json
{
  "0014": "白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪",
  "A001": "你的自定义船只描述"
}
```

设置 `SHIP_DB_PATH=./data/ships.json`，并将 `VECTOR_STORE_AUTO_REBUILD=true` 以重建索引。

## 🧪 测试

```bash
pytest -v                          # 运行全部测试（20 个）
pytest tests/test_database.py -v   # 指定文件
```

## 🛠️ 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| LLM 编排 | LangChain + LangGraph | ReAct Agent 模式 |
| 向量库 | FAISS (faiss-cpu) | 语义检索索引 |
| Embedding | OpenAI Embeddings API | 文本向量化 |
| 配置 | pydantic-settings | 环境变量管理 |
| 向量计算 | NumPy | 余弦相似度 |
| CLI | Rich | 终端美化输出 |
| 测试 | pytest | 单元测试 |

## 📝 开发指南

### 换 LLM Provider

任何兼容 OpenAI API 格式的服务都行：

```env
CHAT_MODEL=gpt-4o
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.openai.com/v1
```

### 添加新工具

在 `tools/__init__.py` 中添加 `@tool` 函数，`build_tools()` 返回列表中加上即可。Agent 会自动识别。

### 数据变更后重建索引

```bash
VECTOR_STORE_AUTO_REBUILD=true ship-hull "触发一次即可重建"
```

或直接删除 `VECTOR_STORE_PATH` 目录，下次启动自动重建。

## 📄 License

MIT
