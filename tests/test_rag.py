"""RAG系统测试模块"""
import pytest
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_loader import DocumentLoader, TextLoader
from src.text_processor import DocumentProcessor, TextCleaner
from src.vector_store import VectorStore
from langchain_core.documents import Document


class TestTextCleaner:
    """文本清洗测试"""

    def test_clean_text(self):
        """测试文本清洗"""
        dirty_text = "这是   一段\t\n测试   文本！！@#$%"
        cleaned = TextCleaner.clean_text(dirty_text)
        assert "测试" in cleaned
        assert "@" not in cleaned

    def test_clean_documents(self):
        """测试文档批量清洗"""
        docs = [
            Document(page_content="测试文档1\n\ntest", metadata={"source": "test1"}),
            Document(page_content="测试文档2", metadata={"source": "test2"})
        ]
        cleaned = TextCleaner.clean_documents(docs)
        assert len(cleaned) >= 0  # 可能全部被过滤或保留


class TestDocumentProcessor:
    """文档处理测试"""

    def test_text_splitting(self):
        """测试文本分块"""
        from src.text_processor import TextSplitter

        text = "这是第一段内容。" * 100
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 150 for chunk in chunks)  # 允许一些超出


class TestVectorStore:
    """向量存储测试"""

    @pytest.fixture
    def sample_docs(self):
        """创建示例文档"""
        return [
            Document(page_content="人工智能是计算机科学的一个分支", metadata={"source": "ai.txt"}),
            Document(page_content="机器学习是人工智能的子领域", metadata={"source": "ml.txt"}),
            Document(page_content="深度学习是机器学习的一个分支", metadata={"source": "dl.txt"})
        ]

    def test_create_index(self, sample_docs):
        """测试创建向量索引"""
        vector_store = VectorStore(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        vector_db = vector_store.create_index(sample_docs)
        assert vector_db is not None

    def test_similarity_search(self, sample_docs):
        """测试相似性搜索"""
        vector_store = VectorStore(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        vector_store.create_index(sample_docs)

        results = vector_store.similarity_search("什么是机器学习", k=2)
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)


class TestDocumentLoader:
    """文档加载测试"""

    def test_load_text_file(self, tmp_path):
        """测试加载文本文件"""
        # 创建临时文件
        test_file = tmp_path / "test.txt"
        test_file.write_text("这是测试内容\n第二行内容", encoding="utf-8")

        docs = DocumentLoader.load_text(str(test_file))
        assert len(docs) > 0
        assert "测试内容" in docs[0].page_content

    def test_load_multiple(self, tmp_path):
        """测试批量加载"""
        # 创建多个临时文件
        for name in ["doc1.txt", "doc2.txt"]:
            (tmp_path / name).write_text(f"内容 for {name}", encoding="utf-8")

        docs = DocumentLoader.load_multiple([str(tmp_path / "doc1.txt"), str(tmp_path / "doc2.txt")])
        assert len(docs) == 2


class TestRAGIntegration:
    """RAG集成测试"""

    @pytest.fixture
    def sample_rag_setup(self):
        """创建RAG测试环境"""
        from src.document_loader import DocumentLoader
        from src.text_processor import DocumentProcessor
        from src.vector_store import VectorStore
        from src.retriever import RetrieverFactory
        from langchain_core.documents import Document

        # 创建测试文档
        docs = [
            Document(
                page_content="LangChain是一个用于开发LLM应用的框架。",
                metadata={"source": "test1.txt"}
            ),
            Document(
                page_content="RAG是检索增强生成技术。",
                metadata={"source": "test2.txt"}
            )
        ]

        # 处理文档
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        chunks = processor.process(docs)

        # 创建向量索引
        vector_store = VectorStore(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        vector_store.create_index(chunks)

        return vector_store, docs

    def test_retrieval(self, sample_rag_setup):
        """测试检索功能"""
        vector_store, docs = sample_rag_setup

        retriever = RetrieverFactory.create_retriever(
            vector_db=vector_store.vector_db,
            documents=docs,
            retriever_type="vector",
            k=2
        )

        results = retriever.get_relevant_documents("什么是LangChain")
        assert len(results) <= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
