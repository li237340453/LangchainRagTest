"""检索器模块 - 配置多种检索策略"""
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
import logging

logger = logging.getLogger(__name__)


class EnsembleRetriever(BaseRetriever):
    """混合检索器 - 结合多个检索器"""

    def __init__(self, retrievers: List[BaseRetriever], weights: List[float] = None):
        self.retrievers = retrievers
        self.weights = weights or [1.0 / len(retrievers)] * len(retrievers)

    def get_relevant_documents(self, query: str) -> List[Document]:
        # 获取所有检索器的结果并合并
        doc_lists = [r.get_relevant_documents(query) for r in self.retrievers]
        return self._merge_documents(doc_lists)

    def _merge_documents(self, doc_lists: List[List[Document]]) -> List[Document]:
        """合并去重文档"""
        seen = set()
        merged = []
        for docs in doc_lists:
            for doc in docs:
                doc_id = doc.page_content[:50]  # 用前50字符作为唯一标识
                if doc_id not in seen:
                    seen.add(doc_id)
                    merged.append(doc)
        return merged


class BaseRetriever:
    """检索器基类"""

    def get_relevant_documents(self, query: str) -> List[Document]:
        """获取相关文档"""
        raise NotImplementedError


class VectorRetriever(BaseRetriever):
    """向量检索器"""

    def __init__(
        self,
        vector_db: FAISS,
        search_type: str = "similarity",
        k: int = 5,
        score_threshold: Optional[float] = None
    ):
        """
        初始化向量检索器

        Args:
            vector_db: FAISS向量数据库
            search_type: 搜索类型 ("similarity", "mmr")
            k: 返回的文档数量
            score_threshold: 分数阈值
        """
        self.vector_db = vector_db
        self.search_type = search_type
        self.k = k
        self.score_threshold = score_threshold

        search_kwargs = {"k": k}
        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold

        self.retriever = vector_db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        logger.info(f"初始化向量检索器, k={k}, search_type={search_type}")

    def get_relevant_documents(self, query: str) -> List[Document]:
        """获取相关文档"""
        return self.retriever.get_relevant_documents(query)


class BM25RetrieverWrapper(BaseRetriever):
    """BM25关键词检索器"""

    def __init__(self, documents: List[Document], k: int = 5):
        """
        初始化BM25检索器

        Args:
            documents: 文档列表
            k: 返回的文档数量
        """
        self.documents = documents
        self.k = k
        self.retriever = BM25Retriever.from_documents(
            documents,
            k=k
        )
        logger.info(f"初始化BM25检索器, k={k}")

    def get_relevant_documents(self, query: str) -> List[Document]:
        """获取相关文档"""
        return self.retriever.get_relevant_documents(query)


class EnsembleRetrieverWrapper(BaseRetriever):
    """混合检索器 - 结合向量检索和BM25"""

    def __init__(
        self,
        vector_db: FAISS,
        documents: List[Document],
        weights: List[float] = [0.7, 0.3],
        k: int = 5
    ):
        """
        初始化混合检索器

        Args:
            vector_db: FAISS向量数据库
            documents: 原始文档列表
            weights: 各检索器权重 [向量权重, BM25权重]
            k: 返回的文档数量
        """
        self.k = k

        # 创建向量检索器
        vector_retriever = vector_db.as_retriever(
            search_kwargs={"k": k}
        )

        # 创建BM25检索器
        bm25_retriever = BM25Retriever.from_documents(
            documents,
            k=k
        )

        # 创建混合检索器
        self.retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=weights
        )
        logger.info(f"初始化混合检索器, weights={weights}, k={k}")

    def get_relevant_documents(self, query: str) -> List[Document]:
        """获取相关文档"""
        return self.retriever.get_relevant_documents(query)


class RetrieverFactory:
    """检索器工厂"""

    @staticmethod
    def create_retriever(
        vector_db: FAISS,
        documents: List[Document],
        retriever_type: str = "vector",
        k: int = 5,
        score_threshold: Optional[float] = None,
        weights: Optional[List[float]] = None
    ) -> BaseRetriever:
        """
        创建检索器

        Args:
            vector_db: FAISS向量数据库
            documents: 文档列表
            retriever_type: 检索器类型 ("vector", "bm25", "ensemble")
            k: 返回的文档数量
            score_threshold: 分数阈值
            weights: 混合检索权重

        Returns:
            检索器实例
        """
        if retriever_type == "vector":
            return VectorRetriever(
                vector_db=vector_db,
                k=k,
                score_threshold=score_threshold
            )
        elif retriever_type == "bm25":
            return BM25RetrieverWrapper(
                documents=documents,
                k=k
            )
        elif retriever_type == "ensemble":
            return EnsembleRetrieverWrapper(
                vector_db=vector_db,
                documents=documents,
                weights=weights or [0.7, 0.3],
                k=k
            )
        else:
            raise ValueError(f"不支持的检索器类型: {retriever_type}")
