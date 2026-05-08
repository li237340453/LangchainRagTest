"""API服务模块 - 使用FastAPI提供RAG服务"""
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import uvicorn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 全局变量（实际使用时通过依赖注入）
_rag_chain = None


def set_rag_chain(chain):
    """设置全局RAG链"""
    global _rag_chain
    _rag_chain = chain


def get_rag_chain():
    """获取全局RAG链"""
    if _rag_chain is None:
        raise HTTPException(status_code=500, detail="RAG链未初始化")
    return _rag_chain


# Pydantic模型
class QueryRequest(BaseModel):
    """查询请求"""
    query: str = Field(..., description="用户问题", min_length=1, max_length=1000)
    top_k: int = Field(5, description="返回的文档数量", ge=1, le=20)
    include_sources: bool = Field(True, description="是否返回源文档")


class QueryResponse(BaseModel):
    """查询响应"""
    answer: str = Field(..., description="AI回答")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="源文档列表")
    query_time: str = Field(..., description="查询时间")
    status: str = Field("success", description="状态")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: str
    version: str = "1.0.0"


# 创建FastAPI应用
app = FastAPI(
    title="RAG API Service",
    description="基于LangChain的检索增强生成API服务",
    version="1.0.0"
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    RAG查询接口

    Args:
        request: 查询请求

    Returns:
        包含回答和源文档的响应
    """
    try:
        rag_chain = get_rag_chain()

        # 执行查询
        result = rag_chain.invoke(request.query)

        # 提取结果
        answer = result.get("result", "")
        source_docs = result.get("source_documents", [])

        # 格式化源文档
        sources = None
        if request.include_sources and source_docs:
            sources = []
            for i, doc in enumerate(source_docs):
                sources.append({
                    "id": i + 1,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                })

        return QueryResponse(
            answer=answer,
            sources=sources,
            query_time=datetime.now().isoformat(),
            status="success"
        )

    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """流式RAG查询接口（返回Server-Sent Events）"""
    from fastapi.responses import StreamingResponse
    import json

    rag_chain = get_rag_chain()

    async def event_generator():
        try:
            # 先检索相关文档
            docs = rag_chain.retriever.get_relevant_documents(request.query)
            yield f"data: {json.dumps({'type': 'retrieval', 'count': len(docs)})}\n\n"

            # 流式生成回答
            for chunk in rag_chain.chain.stream({"query": request.query}):
                if isinstance(chunk, str):
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                elif hasattr(chunk, 'content'):
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk.content})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"流式查询失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    try:
        rag_chain = get_rag_chain()
        return {
            "status": "ready",
            "index_size": len(rag_chain.retriever.vector_db.docstore) if hasattr(rag_chain.retriever, 'vector_db') else 0,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """
    启动API服务器

    Args:
        host: 主机地址
        port: 端口号
        reload: 是否热重载
    """
    logger.info(f"启动RAG API服务: {host}:{port}")
    uvicorn.run(
        "src.api_server:app",
        host=host,
        port=port,
        reload=reload
    )
