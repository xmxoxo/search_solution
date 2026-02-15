import os
from pathlib import Path

# 项目根目录路径
BASE_DIR = Path(__file__).parent.parent

# 日志目录
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 日志级别：DEBUG/INFO/WARNING/ERROR
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
# 日志文件保留天数
LOG_RETENTION_DAYS = 7

# API服务监听地址
API_HOST = os.getenv("API_HOST", "0.0.0.0")
# API服务端口
API_PORT = int(os.getenv("API_PORT", 5310))

# WebUI监听地址
WEBUI_HOST = os.getenv("WEBUI_HOST", "0.0.0.0")
# WebUI端口
WEBUI_PORT = int(os.getenv("WEBUI_PORT", 5320))

# One-API服务密钥 GQtwF5ag8p6m8wWf1232B8D5E17f4455A5C14e7a2d393aEe
ONE_API_KEY = os.getenv("ONE_API_KEY", 'sk-pLLG2ucf61sKFjMxA0Fd11E88c734427A078Bc554e516e26')
# One-API服务地址
ONE_API_BASE_URL = os.getenv("ONE_API_BASE_URL", "http://192.168.15.111:3000/v1")

# Embedding模型名称（用于生成向量） text-embedding-v4  bge-m3:latest
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "bge-m3:latest")
# LLM模型名称（用于查询解析） qwen-max  qwen3:32b
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen-max")

# Redis连接配置
GBL_REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST", "192.168.15.111"),  # Redis主机地址
    "port": int(os.getenv("REDIS_PORT", 6379)),         # Redis端口
    "password": os.getenv("REDIS_PASSWORD", ""),        # Redis密码（可选）
    "db": int(os.getenv("REDIS_DB", 1))                 # Redis数据库编号
}

# SSDB连接地址
SSDB_IP = os.getenv("SSDB_IP", "192.168.15.111")
# SSDB端口
SSDB_PORT = int(os.getenv("SSDB_PORT", 8888))

# Milvus向量数据库主机
MILVUS_HOST = os.getenv("MILVUS_HOST", '192.168.15.111')
# Milvus端口
MILVUS_PORT = int(os.getenv("MILVUS_PORT", 19530))

# MySQL数据库配置
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "192.168.15.111"),    # MySQL主机地址
    "port": int(os.getenv("MYSQL_PORT", 3306)),           # MySQL端口
    "user": os.getenv("MYSQL_USER", "root"),              # MySQL用户名
    "password": os.getenv("MYSQL_PASSWORD", ""),          # MySQL密码
    "database": os.getenv("MYSQL_DATABASE", "ime_db")     # MySQL数据库名
}

# 混合检索中稠密向量的权重（0-1之间，值越大稠密向量权重越高）
HYBRID_SEARCH_ALPHA = float(os.getenv("HYBRID_SEARCH_ALPHA", 0.7))
# 混合检索中稀疏向量的权重（0-1之间，值越大稀疏向量权重越高）
HYBRID_SEARCH_BETA = float(os.getenv("HYBRID_SEARCH_BETA", 0.3))

# 匹配结果最低得分阈值（低于此分数的结果将被过滤）
MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", 0.48))

# 多维度相似度权重配置
MULTI_LEVEL_WEIGHTS = {
    "semantic": float(os.getenv("WEIGHT_SEMANTIC", 0.30)),          # 语义
    "domain": float(os.getenv("WEIGHT_DOMAIN", 0.20)),              # 技术领域
    "method": float(os.getenv("WEIGHT_METHOD", 0.15)),              # 技术方法
    "application": float(os.getenv("WEIGHT_APPLICATION", 0.25)),    # 应用场景
    "innovation": float(os.getenv("WEIGHT_INNOVATION", 0.10))       # 创新点 
}

# 默认返回结果数量
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 10))
# 是否默认启用混合检索
DEFAULT_USE_HYBRID = os.getenv("DEFAULT_USE_HYBRID", "true").lower() == "true"

# LLM解析结果缓存时间（秒，默认24小时）
CACHE_TTL_LLM_PARSE = int(os.getenv("CACHE_TTL_LLM_PARSE", 86400))
# Embedding向量缓存时间（秒，默认7天）
CACHE_TTL_EMBEDDING = int(os.getenv("CACHE_TTL_EMBEDDING", 604800))
# 匹配结果缓存时间（秒，默认30秒）
CACHE_TTL_MATCH_RESULT = int(os.getenv("CACHE_TTL_MATCH_RESULT", 30))

# 中台数据库只读
DATABASE_CONNECT_STRING = "mysql+pymysql://readonly_1633:9a06_qkiLMhQq5T@rm-bp1r9uw2zyl43lrxpdo.mysql.rds.aliyuncs.com:3306/data_middle_group"

# 支持的资源类型列表
RESOURCE_TYPES = [
    "expert",       # 专家
    "project",      # 项目
    "patent",       # 专利
    "enterprise",   # 企业
    "paper",        # 论文
    "institution",  # 高校/科研机构
    "tec"           # 科技成果
]

# 技术成熟度等级（从低到高）
MATURITY_LEVELS = ["研发", "小试", "中试", "量产"]

# 专家职称等级映射（用于标准化处理）
TITLE_LEVELS = {
    "正高": ["正高", "教授", "研究员", "正高级工程师"],
    "副高": ["副高", "副教授", "副研究员", "高级工程师"],
    "中级": ["中级", "讲师", "助理研究员", "工程师"],
    "初级": ["初级", "助教", "实习研究员", "助理工程师"]
}

# 地域名称标准化映射（用于统一地域格式）
REGION_NORMALIZE = {
    "上海": "上海市",
    "北京": "北京市",
    "杭州": "杭州市",
    "深圳": "深圳市",
    "广州": "广州市",
}
