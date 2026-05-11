"""文本处理模块 - 文本清洗和分块"""
import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


class TextCleaner:
    """文本清洗工具"""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        清洗文本内容
        - 去除特殊字符
        - 合并连续空格
        - 去除过长空白行
        """
        # 去除特殊字符（保留中文、英文、数字和常用标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。、！？；：""''（）【】《》,.!?;:"\'\(\)\[\]]', '', text)
        # 合并连续空格
        text = re.sub(r'\s+', ' ', text)
        # 去除首尾空白
        text = text.strip()
        # 去除只有换行符的行
        lines = text.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) > 5]
        text = '\n'.join(lines)
        return text

    @staticmethod
    def clean_documents(documents: List[Document]) -> List[Document]:
        """批量清洗文档"""
        cleaned_docs = []
        for doc in documents:
            cleaned_content = TextCleaner.clean_text(doc.page_content)
            if cleaned_content:
                cleaned_doc = Document(
                    page_content=cleaned_content,
                    metadata=doc.metadata
                )
                cleaned_docs.append(cleaned_doc)
        logger.info(f"清洗完成，共{len(cleaned_docs)}个文档")
        return cleaned_docs


class TextSplitter:
    """文本分块工具"""

    def __init__(
        self,
        chunk_size: int = 150,
        chunk_overlap: int = 30,
        length_function: callable = len
    ):
        """
        初始化文本分割器

        Args:
            chunk_size: 每个文本块的目标大小（字符数）
            chunk_overlap: 相邻块之间的重叠字符数
            length_function: 计算文本长度的函数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档分割成小块"""
        chunks = self.splitter.split_documents(documents)
        logger.info(f"分块完成，共{len(chunks)}个文本块")
        return chunks

    def split_text(self, text: str) -> List[str]:
        """将文本分割成小块"""
        return self.splitter.split_text(text)


class DocumentProcessor:
    """文档处理流程管理器"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        clean: bool = True
    ):
        self.text_cleaner = TextCleaner() if clean else None
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process(self, documents: List[Document]) -> List[Document]:
        """
        完整的文档处理流程：清洗 -> 分块

        Args:
            documents: 原始文档列表

        Returns:
            处理后的文档块列表
        """
        # 清洗
        if self.text_cleaner:
            documents = self.text_cleaner.clean_documents(documents)

        # 分块
        chunks = self.text_splitter.split_documents(documents)

        return chunks
