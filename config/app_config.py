import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_RETENTION_DAYS = 7

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 5310))

WEBUI_HOST = os.getenv("WEBUI_HOST", "0.0.0.0")
WEBUI_PORT = int(os.getenv("WEBUI_PORT", 5320))

ONE_API_KEY = os.getenv("ONE_API_KEY", 'sk-GQtwF5ag8p6m8wWf1232B8D5E17f4455A5C14e7a2d393aEe')
ONE_API_BASE_URL = os.getenv("ONE_API_BASE_URL", "http://192.168.15.111:3000/v1")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "bge-m3:latest")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3")

GBL_REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "192.168.15.111"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "password": os.getenv("REDIS_PASSWORD", ""),
    "db": int(os.getenv("REDIS_DB", 1))
}

SSDB_IP = os.getenv("SSDB_IP", "192.168.15.111")
SSDB_PORT = int(os.getenv("SSDB_PORT", 8888))

MILVUS_HOST = os.getenv("MILVUS_HOST", '192.168.15.111')
MILVUS_PORT = int(os.getenv("MILVUS_PORT", 19530))

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "192.168.15.111"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "ime_db")
}

HYBRID_SEARCH_ALPHA = float(os.getenv("HYBRID_SEARCH_ALPHA", 0.7))
HYBRID_SEARCH_BETA = float(os.getenv("HYBRID_SEARCH_BETA", 0.3))

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 10))
DEFAULT_USE_HYBRID = os.getenv("DEFAULT_USE_HYBRID", "true").lower() == "true"

CACHE_TTL_LLM_PARSE = int(os.getenv("CACHE_TTL_LLM_PARSE", 86400))
CACHE_TTL_EMBEDDING = int(os.getenv("CACHE_TTL_EMBEDDING", 604800))
CACHE_TTL_MATCH_RESULT = int(os.getenv("CACHE_TTL_MATCH_RESULT", 3600))

RESOURCE_TYPES = [
    "expert",
    "project",
    "patent",
    "enterprise",
    "paper",
    "institution",
    "tec"
]

MATURITY_LEVELS = ["研发", "小试", "中试", "量产"]

TITLE_LEVELS = {
    "正高": ["正高", "教授", "研究员", "正高级工程师"],
    "副高": ["副高", "副教授", "副研究员", "高级工程师"],
    "中级": ["中级", "讲师", "助理研究员", "工程师"],
    "初级": ["初级", "助教", "实习研究员", "助理工程师"]
}

REGION_NORMALIZE = {
    "上海": "上海市",
    "北京": "北京市",
    "杭州": "杭州市",
    "深圳": "深圳市",
    "广州": "广州市",
}
