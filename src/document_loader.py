"""文档加载模块 - 支持多种数据源的加载"""
from typing import List, Optional
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    WebBaseLoader,
    CSVLoader,
    Docx2txtLoader
)
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """多源文档加载器"""

    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """加载PDF文件"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"成功加载PDF文件: {file_path}, 共{len(documents)}页")
            return documents
        except Exception as e:
            logger.error(f"加载PDF文件失败: {e}")
            return []

    @staticmethod
    def load_text(file_path: str, encoding: str = "utf-8") -> List[Document]:
        """加载文本文件"""
        try:
            loader = TextLoader(file_path, encoding=encoding)
            documents = loader.load()
            logger.info(f"成功加载文本文件: {file_path}")
            return documents
        except Exception as e:
            logger.error(f"加载文本文件失败: {e}")
            return []

    @staticmethod
    def load_html(file_path: str) -> List[Document]:
        """加载HTML文件"""
        try:
            loader = UnstructuredHTMLLoader(file_path)
            documents = loader.load()
            logger.info(f"成功加载HTML文件: {file_path}")
            return documents
        except Exception as e:
            logger.error(f"加载HTML文件失败: {e}")
            return []

    @staticmethod
    def load_docx(file_path: str) -> List[Document]:
        """加载Word文档"""
        try:
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            logger.info(f"成功加载Word文档: {file_path}")
            return documents
        except Exception as e:
            logger.error(f"加载Word文档失败: {e}")
            return []

    @staticmethod
    def load_csv(file_path: str, encoding: str = "utf-8") -> List[Document]:
        """加载CSV文件"""
        try:
            loader = CSVLoader(file_path, encoding=encoding)
            documents = loader.load()
            logger.info(f"成功加载CSV文件: {file_path}")
            return documents
        except Exception as e:
            logger.error(f"加载CSV文件失败: {e}")
            return []

    @staticmethod
    def load_web(url: str) -> List[Document]:
        """加载网页内容"""
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            logger.info(f"成功加载网页: {url}")
            return documents
        except Exception as e:
            logger.error(f"加载网页失败: {e}")
            return []

    @classmethod
    def load_multiple(cls, file_paths: List[str]) -> List[Document]:
        """批量加载多个文件"""
        all_documents = []
        for file_path in file_paths:
            ext = file_path.lower().split('.')[-1]
            if ext == 'pdf':
                docs = cls.load_pdf(file_path)
            elif ext == 'txt':
                docs = cls.load_text(file_path)
            elif ext in ['html', 'htm']:
                docs = cls.load_html(file_path)
            elif ext == 'docx':
                docs = cls.load_docx(file_path)
            elif ext == 'csv':
                docs = cls.load_csv(file_path)
            else:
                logger.warning(f"不支持的文件类型: {ext}")
                continue
            all_documents.extend(docs)
        return all_documents
