"""LLM模型模块 - 支持本地Ollama和云端API"""
from typing import Optional, List, Dict, Any
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
import logging

logger = logging.getLogger(__name__)


class BaseLLM:
    """LLM基类"""

    def invoke(self, messages: List[BaseMessage]) -> str:
        """调用LLM"""
        raise NotImplementedError

    def generate(self, prompt: str) -> str:
        """生成文本"""
        raise NotImplementedError


class OllamaLLM(BaseLLM):
    """Ollama本地LLM"""

    def __init__(
        self,
        model_name: str = "qwen2:0.5b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.6,
        max_tokens: int = 1000,
        top_p: float = 0.95,
        streaming: bool = False
    ):
        """
        初始化Ollama LLM

        Args:
            model_name: 模型名称
            base_url: Ollama服务地址
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: top_p采样参数
            streaming: 是否启用流式输出
        """
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            streaming=streaming
        )
        logger.info(f"初始化Ollama LLM: {model_name}")

    def invoke(self, messages: List[BaseMessage]) -> str:
        """调用LLM"""
        return self.llm.invoke(messages).content

    def generate(self, prompt: str) -> str:
        """生成文本"""
        messages = [HumanMessage(content=prompt)]
        return self.invoke(messages)


class OpenAILLM(BaseLLM):
    """OpenAI云端LLM"""

    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
        streaming: bool = False,
        callbacks: Optional[List] = None
    ):
        """
        初始化OpenAI LLM

        Args:
            model_name: 模型名称
            api_key: API密钥
            temperature: 温度参数
            max_tokens: 最大token数
            streaming: 是否启用流式输出
            callbacks: 回调函数列表
        """
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            callbacks=callbacks or []
        )
        logger.info(f"初始化OpenAI LLM: {model_name}")

    def invoke(self, messages: List[BaseMessage]) -> str:
        """调用LLM"""
        return self.llm.invoke(messages).content

    def generate(self, prompt: str) -> str:
        """生成文本"""
        messages = [HumanMessage(content=prompt)]
        return self.invoke(messages)


class LLMFactory:
    """LLM工厂类"""

    @staticmethod
    def create_llm(
        llm_type: str = "ollama",
        **kwargs
    ) -> BaseLLM:
        """
        创建LLM实例

        Args:
            llm_type: LLM类型 ("ollama", "openai")
            **kwargs: 其他参数

        Returns:
            LLM实例
        """
        if llm_type == "ollama":
            return OllamaLLM(
                model_name=kwargs.get("model_name", "qwen2:0.5b"),
                base_url=kwargs.get("base_url", "http://localhost:11434"),
                temperature=kwargs.get("temperature", 0.6),
                max_tokens=kwargs.get("max_tokens", 1000),
                streaming=kwargs.get("streaming", False)
            )
        elif llm_type == "openai":
            return OpenAILLM(
                model_name=kwargs.get("model_name", "gpt-4-turbo"),
                api_key=kwargs.get("api_key"),
                temperature=kwargs.get("temperature", 0.2),
                max_tokens=kwargs.get("max_tokens", 1000),
                streaming=kwargs.get("streaming", False),
                callbacks=kwargs.get("callbacks", [])
            )
        else:
            raise ValueError(f"不支持的LLM类型: {llm_type}")
