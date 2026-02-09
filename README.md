# 智能匹配引擎 (Intelligent Matching Engine)

一个基于语义理解的智能匹配系统，支持成果/专利/项目相似性检索、需求与多类型资源匹配。

## 功能特性

- **语义匹配**：基于BGE-M3向量模型的语义相似度计算
- **混合检索**：稠密向量 + 稀疏向量融合检索
- **多资源类型**：支持专家、项目、专利、企业、论文、高校、成果7类资源
- **条件过滤**：支持地域、成熟度等多维度条件过滤
- **LLM查询理解**：自动从自然语言提取意图和结构化条件

## 技术栈

- **后端**: FastAPI + Python 3.11
- **前端**: Streamlit
- **向量数据库**: Milvus 2.5.0
- **缓存**: Redis
- **LLM**: Qwen3 (通过One-API)
- **Embedding**: BGE-M3

## 快速开始

### 环境要求

- Python 3.11+
- Milvus 2.5.0
- Redis

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置

编辑 `config/app_config.py` 修改相关配置：
- One-API地址和密钥
- Redis连接信息
- Milvus连接信息

### 启动服务

**Windows环境：**

```bash
# 启动API服务
start.cmd api

# 启动WebUI（另一个终端）
start.cmd webui

# 或同时启动
start.cmd all
```

**Linux/Mac环境：**

```bash
# 启动API服务
./start.sh api

# 启动WebUI（另一个终端）
./start.sh webui

# 或同时启动
./start.sh all
```

### Docker部署

```bash
# 构建镜像
docker-compose build

# 启动容器
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止容器
docker-compose down
```

## 访问地址

- **API服务**: http://localhost:5310
- **API文档**: http://localhost:5310/docs
- **WebUI**: http://localhost:5320

## API接口

### 1. 数据匹配接口

```bash
POST /api/v1/match
Content-Type: application/json

{
  "query": "找上海做AI大模型的专家",
  "resource_types": ["expert", "project"],
  "top_k": 10,
  "use_hybrid": true
}
```

### 2. 数据导入接口

```bash
POST /api/v1/data/insert
Content-Type: application/json

{
  "resource_type": "expert",
  "items": [
    {
      "id": "exp_001",
      "fields": {
        "name": "张三",
        "research_direction": "人工智能"
      },
      "raw_text_for_embedding": "张三教授研究方向为人工智能..."
    }
  ]
}
```

### 3. 查询解析接口

```bash
POST /api/v1/parse-query
Content-Type: application/json

{
  "query": "找杭州做生物医药的企业"
}
```

## 项目结构

```
search_solution/
├── api/                    # FastAPI应用
│   ├── main.py            # 主应用入口
│   ├── core/              # 核心业务逻辑
│   ├── routers/           # 路由模块
│   ├── models/            # 数据模型
│   └── utils/             # 工具函数
├── webui/                 # Streamlit应用
│   └── main.py           # WebUI入口
├── config/                # 配置文件
├── docs/                  # 文档
├── tests/                 # 测试代码
├── logs/                  # 日志目录
├── Dockerfile             # Docker构建
├── docker-compose.yml     # Docker编排
├── start.sh               # Linux启动脚本
├── start.cmd              # Windows启动脚本
└── requirements.txt       # Python依赖
```

## 资源类型

| 类型 | 说明 |
|------|------|
| expert | 专家 |
| project | 项目 |
| patent | 专利 |
| enterprise | 企业 |
| paper | 论文 |
| institution | 高校 |
| tec | 成果 |

## 开发日志

详细开发记录请查看 [docs/dev.md](docs/dev.md)

## 许可证

MIT
