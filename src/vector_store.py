"""向量存储模块 - 使用FAISS进行向量索引和存储"""
from typing import List, Optional, Union
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import OpenAIEmbeddings
import logging
import os

logger = logging.getLogger(__name__)


class VectorStore:
    """向量存储管理器"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = "cpu",
        encode_kwargs: dict = None,
        embedding_type: str = "huggingface",
        api_key: str = None,
        base_url: str = None
    ):
        """
        初始化向量存储

        Args:
            model_name: Embedding模型名称
            device: 设备类型 ("cpu" 或 "cuda")
            encode_kwargs: 编码参数
            embedding_type: Embedding类型 ("huggingface" 或 "siliconflow")
            api_key: API密钥 (用于siliconflow)
            base_url: API基础URL (用于siliconflow)
        """
        if embedding_type == "siliconflow":
            # 使用SiliconFlow Embedding API (OpenAI兼容)
            if base_url is None:
                base_url = "https://api.siliconflow.cn/v1"
            self.embeddings = OpenAIEmbeddings(
                model=model_name,
                api_key=api_key,
                base_url=base_url
            )
            self.embedding_type = "siliconflow"
            logger.info(f"初始化SiliconFlow Embedding模型: {model_name}")
        else:
            # 使用HuggingFace本地模型
            if encode_kwargs is None:
                encode_kwargs = {"batch_size": 32, "show_progress_bar": False}
            self.embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs=encode_kwargs
            )
            self.embedding_type = "huggingface"
            logger.info(f"初始化HuggingFace Embedding模型: {model_name} on {device}")

        self.vector_db = None

    def create_index(self, documents: List[Document], index_name: str = "faiss_index", batch_size: int = 32) -> FAISS:
        """
        从文档创建向量索引

        Args:
            documents: 文档列表
            index_name: 索引名称（用于保存）
            batch_size: 批处理大小（SiliconFlow限制最大32）

        Returns:
            FAISS向量数据库对象
        """
        if self.embedding_type == "siliconflow":
            # SiliconFlow API: 分批创建索引再合并
            first_batch = documents[:batch_size]
            self.vector_db = FAISS.from_documents(documents=first_batch, embedding=self.embeddings)
            logger.info(f"已处理 {len(first_batch)}/{len(documents)} 个文档块")
            
            for i in range(batch_size, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_db = FAISS.from_documents(documents=batch, embedding=self.embeddings)
                self.vector_db.merge_from(batch_db)
                logger.info(f"已处理 {min(i + batch_size, len(documents))}/{len(documents)} 个文档块")
        else:
            self.vector_db = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
        logger.info(f"创建向量索引，包含{len(documents)}个文档块")
        return self.vector_db

    def save_index(self, index_path: str):
        """
        保存向量索引到本地

        Args:
            index_path: 保存路径
        """
        if self.vector_db is None:
            raise ValueError("向量索引未创建，请先调用create_index()")
        self.vector_db.save_local(index_path)
        logger.info(f"向量索引已保存到: {index_path}")

    def load_index(self, index_path: str, embeddings=None) -> FAISS:
        """
        从本地加载向量索引

        Args:
            index_path: 索引路径
            embeddings: 嵌入模型（如果不传则使用初始化时的模型）

        Returns:
            FAISS向量数据库对象
        """
        if embeddings is None:
            embeddings = self.embeddings
        self.vector_db = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"从{index_path}加载向量索引")
        return self.vector_db

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        相似性搜索

        Args:
            query: 查询文本
            k: 返回的最近邻数量
            score_threshold: 相似度分数阈值（可选）

        Returns:
            相似文档列表
        """
        if self.vector_db is None:
            raise ValueError("向量索引未加载，请先调用create_index()或load_index()")

        search_kwargs = {"k": k}
        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold

        results = self.vector_db.similarity_search(
            query,
            **search_kwargs
        )
        logger.info(f"检索到{len(results)}个相关文档")
        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5
    ) -> List[tuple]:
        """
        带分数的相似性搜索

        Args:
            query: 查询文本
            k: 返回的最近邻数量

        Returns:
            (文档, 分数)元组列表
        """
        if self.vector_db is None:
            raise ValueError("向量索引未加载，请先调用create_index()或load_index()")

        results = self.vector_db.similarity_search_with_score(query, k=k)
        logger.info(f"检索到{len(results)}个相关文档")
        return results

    def add_documents(self, documents: List[Document]):
        """向现有索引添加新文档"""
        if self.vector_db is None:
            raise ValueError("向量索引未加载")
        self.vector_db.add_documents(documents)
        logger.info(f"添加{len(documents)}个文档到索引")

    def merge_from(self, vector_db: FAISS):
        """合并另一个向量数据库"""
        if self.vector_db is None:
            self.vector_db = vector_db
        else:
            self.vector_db.merge_from(vector_db)
        logger.info("向量索引合并完成")

    @property
    def docstore(self):
        """获取文档存储"""
        return self.vector_db.docstore if self.vector_db else None

    @property
    def index(self):
        """获取FAISS索引"""
        return self.vector_db.index if self.vector_db else None
