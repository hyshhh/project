# 🚢 Ship Hull Agent — 船弦号识别 Agent

基于 LangChain + LangGraph 的智能船弦号识别系统，支持 **精确弦号匹配** 和 **语义相似度检索** 两种识别模式。

## ✨ 功能特性

- **精确匹配**：输入弦号直接查数据库，毫秒级响应
- **语义检索**：弦号不存在时，使用 Embedding 模型对船只外观描述做余弦相似度匹配
- **可扩展数据库**：支持内置数据和外部 JSON 文件加载
- **CLI 工具**：命令行单次查询 & 交互式 REPL
- **模块化架构**：配置 / 数据库 / 工具 / Agent 各层解耦，易于二次开发

## 🏗️ 项目结构

```
ship-hull-agent/
├── config/
│   └── __init__.py          # pydantic-settings 配置管理，支持 .env
├── database/
│   └── __init__.py          # ShipDatabase（精确查找）+ EmbeddingIndex（语义检索）
├── tools/
│   └── __init__.py          # LangChain @tool 定义（lookup / search）
├── agent/
│   └── __init__.py          # ShipHullAgent 类 + create_agent() 工厂
├── cli/
│   ├── __init__.py          # Rich CLI，单次查询 & 交互 REPL
│   └── main.py              # python -m cli.main 入口
├── tests/
│   ├── test_database.py     # 数据库单元测试（8 个）
│   └── test_config.py       # 配置单元测试（4 个）
├── .env.example             # 环境变量模板
├── .gitignore
├── pyproject.toml           # 项目元数据 + 依赖声明
└── README.md
```

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

# Embedding 模型（阿里云 DashScope）
EMBED_MODEL=text-embedding-v4
EMBED_API_KEY=your-embed-api-key
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 可选：自定义数据库路径
# SHIP_DB_PATH=./data/ships.json
```

### 4. 运行

```bash
# 单次查询
ship-hull "帮我查一下弦号0014是什么船"

# 交互模式
ship-hull --interactive

# 或者直接 python 运行
python -m cli.main "弦号0256是什么船"
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

### 纯描述检索（无弦号）

```
$ ship-hull "我看到一艘灰色的军舰，外形很隐身，船头有导弹发射装置"

未找到对应弦号，根据描述检索到最相似的船：弦号 0256，描述：灰色军舰，隐身外形设计，舰首配有垂直发射系统（相似度：0.8912）
```

### 作为 Python 库调用

```python
from agent import create_agent

agent = create_agent()

# 单次查询
answer = agent.run("弦号0014是什么船")
print(answer)
```

## ⚙️ 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CHAT_MODEL` | 对话模型名称 | `Qwen/Qwen3-VL-4B-AWQ` |
| `LLM_API_KEY` | 对话模型 API Key | `abc123` |
| `LLM_BASE_URL` | 对话模型地址 | `http://localhost:7890/v1` |
| `EMBED_MODEL` | Embedding 模型名称 | `text-embedding-v4` |
| `EMBED_API_KEY` | Embedding API Key | — |
| `EMBED_BASE_URL` | Embedding 服务地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `SHIP_DB_PATH` | 自定义数据库 JSON 路径 | 内置默认数据 |
| `LOG_LEVEL` | 日志级别 | `INFO` |

### 自定义数据库

创建一个 JSON 文件（如 `data/ships.json`），格式为 `{弦号: 描述}` 的映射：

```json
{
  "0014": "白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪",
  "0025": "黑色散货船，船体有红色水线，甲板上配有龙门吊",
  "A001": "你的自定义船只描述"
}
```

然后设置环境变量：

```bash
export SHIP_DB_PATH=./data/ships.json
```

## 🧪 测试

```bash
# 运行全部测试
pytest

# 运行指定测试文件
pytest tests/test_database.py -v

# 测试覆盖率
pytest --cov=. --cov-report=term-missing
```

## 🔧 Agent 工作流程

```
用户输入
  │
  ├─ 包含弦号？ ──→ lookup_by_hull_number() ──→ 精确匹配成功？ ──→ 返回结果
  │                                         └─ 否 ──→ search_by_description()
  │
  └─ 仅描述？ ──→ search_by_description() ──→ 语义检索 Top-1 ──→ 返回结果
```

**工具说明：**

| 工具 | 用途 | 调用时机 |
|------|------|---------|
| `lookup_by_hull_number` | 精确弦号查找 | 用户提供了弦号时优先调用 |
| `search_by_description` | Embedding 语义检索 | 弦号查不到 / 用户仅提供描述 |

## 🛠️ 技术栈

- **LangChain** — LLM 编排框架
- **LangGraph** — Agent 状态图（ReAct 模式）
- **pydantic-settings** — 配置管理
- **NumPy** — 向量计算（余弦相似度）
- **Rich** — 终端美化输出
- **pytest** — 测试框架

## 📝 开发指南

### 添加新的 LLM Provider

编辑 `.env`，修改 `LLM_BASE_URL` 和 `LLM_MODEL` 即可，只要兼容 OpenAI API 格式：

```env
# 例如使用 OpenAI
CHAT_MODEL=gpt-4o
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.openai.com/v1
```

### 扩展 Embedding Provider

编辑 `database/__init__.py` 中的 `EmbeddingIndex` 类，替换为你需要的 Embedding 客户端。

### 添加新工具

在 `tools/__init__.py` 中添加新的 `@tool` 函数，然后在 `build_tools()` 中返回即可。Agent 会自动识别并调用。

## 📄 License

MIT
