"""RAG链模块 - 使用LCEL构建检索增强生成链"""
from typing import Dict, Any, Optional, List
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import logging

logger = logging.getLogger(__name__)


# 默认提示模板
DEFAULT_PROMPT_TEMPLATE = """已知信息：
{context}

用户问题：
{question}

请基于已知信息回答用户问题。如果已知信息中没有相关内容，请明确告知用户。
"""


def format_docs(docs: List) -> str:
    """格式化文档为上下文字符串"""
    return "\n\n".join(doc.page_content for doc in docs)


class SimpleMemory:
    """简单的对话记忆"""

    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def save_context(self, inputs: Dict, outputs: Dict):
        self.history.append({"user": inputs.get("question", ""), "assistant": outputs.get("answer", "")})

    def load_memory_variables(self, _) -> Dict[str, Any]:
        chat_history = "\n".join([f"用户: {h['user']}\n助手: {h['assistant']}" for h in self.history])
        return {"chat_history": chat_history}

    def clear(self):
        self.history = []


class RAGChain:
    """RAG链管理器 - 使用LCEL"""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLanguageModel,
        prompt_template: Optional[str] = None,
        verbose: bool = True,
        return_source_documents: bool = True
    ):
        self.retriever = retriever
        self.llm = llm
        self.return_source_documents = return_source_documents

        if prompt_template is None:
            prompt_template = DEFAULT_PROMPT_TEMPLATE

        prompt = PromptTemplate.from_template(prompt_template)

        self.chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("初始化RAG链完成")

    def invoke(self, query: str, **kwargs) -> Dict[str, Any]:
        try:
            result = self.chain.invoke(query, **kwargs)
            return {"result": result, "source_documents": []}
        except Exception as e:
            logger.error(f"RAG查询失败: {e}")
            return {"result": f"查询失败: {str(e)}", "source_documents": []}

    def __call__(self, query: str) -> str:
        result = self.invoke(query)
        return result.get("result", "")


class ConversationalRAGChain(RAGChain):
    """带对话记忆的RAG链"""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLanguageModel,
        **kwargs
    ):
        super().__init__(retriever, llm, **kwargs)
        self.memory = SimpleMemory()

        prompt = PromptTemplate.from_template(
            DEFAULT_PROMPT_TEMPLATE + "\n\n历史对话：\n{chat_history}"
        )

        def load_memory(_):
            return self.memory.load_memory_variables({})["chat_history"]

        self.chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": load_memory
            }
            | prompt
            | llm
            | StrOutputParser()
        )

    def invoke(self, query: str, **kwargs) -> Dict[str, Any]:
        result = super().invoke(query, **kwargs)
        self.memory.save_context({"question": query}, {"answer": result.get("result", "")})
        return result


class RAGChainBuilder:
    """RAG链构建器"""

    @staticmethod
    def build_simple_rag(
        retriever: BaseRetriever,
        llm: BaseLanguageModel,
        custom_prompt: Optional[str] = None
    ) -> RAGChain:
        return RAGChain(
            retriever=retriever,
            llm=llm,
            prompt_template=custom_prompt
        )

    @staticmethod
    def build_conversational_rag(
        retriever: BaseRetriever,
        llm: BaseLanguageModel
    ) -> ConversationalRAGChain:
        return ConversationalRAGChain(
            retriever=retriever,
            llm=llm
        )
