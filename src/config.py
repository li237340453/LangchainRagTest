# 配置模块
import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# 索引存储目录
INDEX_DIR = BASE_DIR / "index"
INDEX_DIR.mkdir(exist_ok=True)

# 模型配置
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
EMBEDDING_DEVICE = "cpu"  # 可选: "cpu", "cuda"

# LLM配置 (使用Ollama本地模型)
LLM_MODEL_NAME = "qwen2.5:14b"  # 可根据实际部署的模型修改
LLM_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.6
LLM_MAX_TOKENS = 1000

# 文本分块配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 检索配置
DEFAULT_TOP_K = 5
SCORE_THRESHOLD = 0.7

# API服务配置
API_HOST = "0.0.0.0"
API_PORT = 8000
