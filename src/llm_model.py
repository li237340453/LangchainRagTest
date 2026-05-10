"""LLM模型模块 - 支持本地Ollama和云端API"""
from typing import Optional, List, Dict, Any, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import GenerationChunk, ChatResult, ChatGeneration
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field
import logging

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLanguageModel):
    """Ollama本地LLM"""

    model_name: str = Field(default="qwen2.5:14b", alias="model_name")
    base_url: str = Field(default="http://localhost:11434")
    temperature: float = Field(default=0.6)
    max_tokens: int = Field(default=1000)
    top_p: float = Field(default=0.95)
    streaming: bool = Field(default=False)

    def __init__(self, **data):
        # 处理别名
        if "model" in data:
            data["model_name"] = data.pop("model")
        super().__init__(**data)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from langchain_community.chat_models import ChatOllama
            self._client = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                streaming=self.streaming
            )
        return self._client

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        client = self._get_client()
        response = client.invoke(prompt)
        return response if isinstance(response, str) else response.content

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        client = self._get_client()
        results = []
        for prompt in prompts:
            response = client.invoke(prompt)
            content = response if isinstance(response, str) else response.content
            results.append(ChatGeneration(text=content))
        return ChatResult(generations=results)

    def invoke(self, input: Union[str, List[BaseMessage]], config=None, **kwargs) -> BaseMessage:
        client = self._get_client()
        if config is not None:
            return client.invoke(input, config=config, **kwargs)
        return client.invoke(input, **kwargs)

    def batch(self, inputs: List, **kwargs) -> List:
        return [self.invoke(inp, **kwargs) for inp in inputs]

    def generate_prompt(self, prompts: List[str], **kwargs) -> ChatResult:
        return self._generate(prompts, **kwargs)

    async def agenerate_prompt(self, prompts: List[str], **kwargs) -> ChatResult:
        return self.generate_prompt(prompts, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "ollama_llm"


class OpenAILLM(BaseLanguageModel):
    """OpenAI云端LLM"""

    model_name: str = Field(default="gpt-4-turbo", alias="model_name")
    api_key: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=1000)
    streaming: bool = Field(default=False)

    def __init__(self, **data):
        if "model" in data:
            data["model_name"] = data.pop("model")
        super().__init__(**data)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from langchain_openai import ChatOpenAI
            self._client = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                streaming=self.streaming
            )
        return self._client

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        client = self._get_client()
        response = client.invoke(prompt)
        return response if isinstance(response, str) else response.content

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        client = self._get_client()
        results = []
        for prompt in prompts:
            response = client.invoke(prompt)
            content = response if isinstance(response, str) else response.content
            results.append(ChatGeneration(text=content))
        return ChatResult(generations=results)

    def invoke(self, input: Union[str, List[BaseMessage]], config=None, **kwargs) -> BaseMessage:
        client = self._get_client()
        if config is not None:
            return client.invoke(input, config=config, **kwargs)
        return client.invoke(input, **kwargs)

    def batch(self, inputs: List, **kwargs) -> List:
        return [self.invoke(inp, **kwargs) for inp in inputs]

    def generate_prompt(self, prompts: List[str], **kwargs) -> ChatResult:
        return self._generate(prompts, **kwargs)

    async def agenerate_prompt(self, prompts: List[str], **kwargs) -> ChatResult:
        return self.generate_prompt(prompts, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "openai_llm"


class LLMFactory:
    """LLM工厂类"""

    @staticmethod
    def create_llm(llm_type: str = "ollama", **kwargs):
        if llm_type == "ollama":
            return OllamaLLM(
                model_name=kwargs.get("model_name", "qwen2:0.5b"),
                base_url=kwargs.get("base_url", "http://localhost:11434"),
                temperature=kwargs.get("temperature", 0.6),
                max_tokens=kwargs.get("max_tokens", 1000)
            )
        elif llm_type == "openai":
            return OpenAILLM(
                model_name=kwargs.get("model_name", "gpt-4-turbo"),
                api_key=kwargs.get("api_key"),
                temperature=kwargs.get("temperature", 0.2),
                max_tokens=kwargs.get("max_tokens", 1000)
            )
        else:
            raise ValueError(f"不支持的LLM类型: {llm_type}")
