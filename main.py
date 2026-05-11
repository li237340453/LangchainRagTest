"""
RAG系统主入口
提供命令行交互和API服务两种模式
"""
import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    DATA_DIR, INDEX_DIR, EMBEDDING_MODEL, EMBEDDING_DEVICE,
    CHUNK_SIZE, CHUNK_OVERLAP, LLM_MODEL_NAME, LLM_BASE_URL,
    LLM_API_KEY, LLM_TEMPERATURE, SILICONFLOW_API_KEY,
    SILICONFLOW_MODEL_NAME, SILICONFLOW_BASE_URL,
    SILICONFLOW_EMBEDDING_MODEL,
    DEFAULT_TOP_K, API_HOST, API_PORT
)
from src.document_loader import DocumentLoader
from src.text_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.retriever import RetrieverFactory
from src.llm_model import LLMFactory
from src.rag_chain import RAGChain, RAGChainBuilder
from src.api_server import set_rag_chain, run_server

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG系统管理器"""

    def __init__(self):
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.rag_chain = None
        self.documents = []

    def load_documents(self, data_dir: Path = None):
        """加载文档"""
        if data_dir is None:
            data_dir = DATA_DIR

        # 支持的文件格式
        file_patterns = ["*.txt", "*.pdf", "*.docx", "*.html"]
        file_paths = []
        for pattern in file_patterns:
            file_paths.extend(list(data_dir.glob(pattern)))

        if not file_paths:
            logger.warning(f"在 {data_dir} 中未找到文档文件")
            return self

        # 加载文档
        all_docs = []
        for file_path in file_paths:
            logger.info(f"加载文件: {file_path.name}")
            docs = DocumentLoader.load_multiple([str(file_path)])
            all_docs.extend(docs)

        self.documents = all_docs
        logger.info(f"共加载 {len(self.documents)} 个文档")
        return self

    def build_index(self, index_dir: Path = None, force_rebuild: bool = False,
                    embedding_type: str = "siliconflow", api_key: str = None):
        """构建向量索引"""
        if index_dir is None:
            index_dir = INDEX_DIR

        index_path = index_dir / "faiss_index"

        # 获取API密钥
        api_key = api_key or SILICONFLOW_API_KEY

        # 获取模型名称
        if embedding_type == "siliconflow":
            embedding_model = SILICONFLOW_EMBEDDING_MODEL
        else:
            embedding_model = EMBEDDING_MODEL

        # 检查是否已有索引
        if index_path.exists() and not force_rebuild:
            logger.info(f"从本地加载索引: {index_path}")
            self.vector_store = VectorStore(
                model_name=embedding_model,
                device=EMBEDDING_DEVICE,
                embedding_type=embedding_type,
                api_key=api_key,
                base_url=SILICONFLOW_BASE_URL
            )
            self.vector_store.load_index(str(index_path))
        else:
            if not self.documents:
                logger.error("没有可用的文档，请先加载文档")
                return self

            logger.info("创建向量索引...")

            # 处理文档：清洗和分块
            processor = DocumentProcessor(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                clean=True
            )
            chunks = processor.process(self.documents)

            # 创建向量索引
            self.vector_store = VectorStore(
                model_name=embedding_model,
                device=EMBEDDING_DEVICE,
                embedding_type=embedding_type,
                api_key=api_key,
                base_url=SILICONFLOW_BASE_URL
            )
            self.vector_store.create_index(chunks)

            # 保存索引
            self.vector_store.save_index(str(index_path))
            logger.info(f"索引已保存到: {index_path}")

        return self

    def setup_retriever(self, retriever_type: str = "vector", k: int = None):
        """设置检索器"""
        if self.vector_store is None:
            logger.error("向量索引未创建")
            return self

        if k is None:
            k = DEFAULT_TOP_K

        self.retriever = RetrieverFactory.create_retriever(
            vector_db=self.vector_store.vector_db,
            documents=self.documents,
            retriever_type=retriever_type,
            k=k
        )
        logger.info(f"检索器已设置: {retriever_type}, k={k}")
        return self

    def setup_llm(self, llm_type: str = "siliconflow", api_key: str = None):
        """设置LLM"""
        # 硅基流动配置
        if llm_type == "siliconflow":
            api_key = api_key or SILICONFLOW_API_KEY
            if not api_key:
                logger.error("使用硅基流动需要提供API密钥，请设置SILICONFLOW_API_KEY或--api-key参数")
                return self
            self.llm = LLMFactory.create_siliconflow_llm(
                api_key=api_key,
                model_name=SILICONFLOW_MODEL_NAME,
                temperature=LLM_TEMPERATURE
            )
        else:
            self.llm = LLMFactory.create_llm(
                llm_type=llm_type,
                model_name=LLM_MODEL_NAME,
                base_url=LLM_BASE_URL,
                temperature=LLM_TEMPERATURE,
                api_key=api_key
            )
        logger.info(f"LLM已设置: {llm_type}")
        return self

    def build_chain(self):
        """构建RAG链"""
        if self.retriever is None or self.llm is None:
            logger.error("检索器或LLM未设置")
            return self

        self.rag_chain = RAGChainBuilder.build_simple_rag(
            retriever=self.retriever,
            llm=self.llm
        )
        logger.info("RAG链已构建")
        return self

    def query(self, question: str):
        """执行查询"""
        if self.rag_chain is None:
            logger.error("RAG链未构建")
            return None

        result = self.rag_chain.invoke(question)
        return result

    def interactive_mode(self):
        """交互模式"""
        print("\n" + "=" * 50)
        print("RAG系统交互模式")
        print("=" * 50)
        print("输入问题进行查询，输入 'quit' 或 'exit' 退出")
        print("-" * 50)

        while True:
            try:
                question = input("\n用户: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("再见!")
                    break

                if not question:
                    continue

                print("\n检索中...")
                result = self.query(question)

                if result:
                    print("\n" + "-" * 50)
                    print("AI回答:")
                    print(result.get("result", ""))
                    print("-" * 50)

                    sources = result.get("source_documents", [])
                    if sources:
                        print(f"\n参考文档 ({len(sources)}个):")
                        for i, doc in enumerate(sources[:3], 1):
                            content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                            print(f"  {i}. {content}")

            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as e:
                logger.error(f"查询出错: {e}")
                print(f"查询出错: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG系统 - 基于LangChain的检索增强生成")
    parser.add_argument("--mode", choices=["interactive", "api", "index"], default="interactive",
                        help="运行模式: interactive(交互), api(API服务), index(仅构建索引)")
    parser.add_argument("--data-dir", type=str, default=None, help="数据目录路径")
    parser.add_argument("--index-dir", type=str, default=None, help="索引保存目录")
    parser.add_argument("--rebuild-index", action="store_true", help="强制重建索引")
    parser.add_argument("--llm-type", choices=["ollama", "openai", "siliconflow"], default="siliconflow", help="LLM类型")
    parser.add_argument("--embedding-type", choices=["huggingface", "siliconflow"], default="siliconflow",
                        help="Embedding类型: huggingface(本地) 或 siliconflow(云端)")
    parser.add_argument("--api-key", type=str, default=None, help="API密钥 (硅基流动或OpenAI)")
    parser.add_argument("--retriever-type", choices=["vector", "bm25", "ensemble"], default="vector",
                        help="检索器类型")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="检索返回数量")
    parser.add_argument("--host", type=str, default=API_HOST, help="API服务主机")
    parser.add_argument("--port", type=int, default=API_PORT, help="API服务端口")

    args = parser.parse_args()

    # 初始化RAG系统
    rag_system = RAGSystem()

    # 根据模式执行
    if args.mode in ["interactive", "index"]:
        data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
        index_dir = Path(args.index_dir) if args.index_dir else INDEX_DIR

        # 加载文档
        rag_system.load_documents(data_dir)

        if not rag_system.documents:
            logger.error("没有加载到任何文档，请检查数据目录")
            sys.exit(1)

        # 构建索引
        rag_system.build_index(index_dir, force_rebuild=args.rebuild_index,
                               embedding_type=args.embedding_type, api_key=args.api_key)

        # 设置检索器
        rag_system.setup_retriever(retriever_type=args.retriever_type, k=args.k)

        # 设置LLM
        rag_system.setup_llm(llm_type=args.llm_type, api_key=args.api_key)

        if args.mode == "interactive":
            # 构建RAG链
            rag_system.build_chain()

            # 启动交互模式
            rag_system.interactive_mode()

    elif args.mode == "api":
        # API模式需要在启动时初始化完整的RAG链
        logger.info("初始化RAG系统...")

        data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
        index_dir = Path(args.index_dir) if args.index_dir else INDEX_DIR

        rag_system.load_documents(data_dir)
        rag_system.build_index(index_dir, force_rebuild=args.rebuild_index,
                               embedding_type=args.embedding_type, api_key=args.api_key)
        rag_system.setup_retriever(retriever_type=args.retriever_type, k=args.k)
        rag_system.setup_llm(llm_type=args.llm_type, api_key=args.api_key)
        rag_system.build_chain()

        # 设置全局RAG链
        set_rag_chain(rag_system.rag_chain)

        # 启动API服务
        logger.info(f"启动API服务: http://{args.host}:{args.port}")
        run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
