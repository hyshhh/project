"""配置模块测试"""

from config import LLMConfig, EmbedConfig, AppConfig


class TestLLMConfig:
    def test_defaults(self):
        c = LLMConfig()
        assert c.model == "Qwen/Qwen3-VL-4B-AWQ"
        assert c.api_key == "abc123"

    def test_override(self):
        c = LLMConfig(CHAT_MODEL="gpt-4o")
        assert c.model == "gpt-4o"


class TestEmbedConfig:
    def test_defaults(self):
        c = EmbedConfig()
        assert c.model == "text-embedding-v4"
        assert "dashscope" in c.base_url


class TestAppConfig:
    def test_defaults(self):
        c = AppConfig()
        assert c.log_level == "INFO"
        assert c.ship_db_path is None
        assert isinstance(c.llm, LLMConfig)
        assert isinstance(c.embed, EmbedConfig)
