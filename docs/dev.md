# 开发日志

**创建时间**: 2026-02-09  
**项目名称**: 智能匹配引擎（Intelligent Matching Engine, IME）

---

## 一、项目概述

本项目构建一个通用的智能匹配模块，支持三类核心场景：
1. **相似性匹配**：成果/专利/项目之间的语义相似性检索
2. **需求-资源匹配**：自然语言需求与多类型资源（成果/专家/项目/专利/企业/论文/高校）的智能匹配
3. **关键词增强搜索**：结合知识图谱的语义检索

---

## 二、技术栈确认

| 类别 | 技术选型 | 说明 |
|------|---------|------|
| 开发语言 | Python | - |
| 基础库 | baselibs 0.1.15 | RedisCache、LLM_TASK 组件 |
| API框架 | FastAPI | 端口：5310 |
| WebUI | Streamlit | 端口：5320 |
| 数据库 | SSDB | 简单数据存储 |
| 关系数据库 | MySQL 8 | 复杂元数据存储 |
| 向量数据库 | Milvus 2.5.0 | 支持混合检索（dense + sparse） |
| 缓存 | Redis | 缓存LLM解析、向量、匹配结果 |
| LLM | Qwen3 系列 | 通过 One-API 调用 |
| Embedding | BGE-M3 | 输出 dense + sparse 向量 |

---

## 三、目录结构规划

```
search_solution/
├── api/                          # FastAPI 应用
│   ├── main.py                   # 入口文件
│   ├── routers/                  # 路由模块
│   │   ├── __init__.py
│   │   ├── match.py              # 匹配接口
│   │   ├── data.py               # 数据导入接口
│   │   └── parse.py              # 查询解析接口
│   ├── core/                     # 核心业务逻辑
│   │   ├── __init__.py
│   │   ├── llm_parser.py         # LLM 查询解析
│   │   ├── embedding.py          # 向量生成（BGE-M3）
│   │   ├── milvus_search.py      # Milvus 混合检索
│   │   └── matcher.py            # 匹配逻辑与评分
│   ├── models/                   # 数据模型
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic 模型
│   └── utils/                    # 工具函数
│       ├── __init__.py
│       └── cache.py              # Redis 缓存封装
├── webui/                        # Streamlit 应用
│   ├── main.py                   # 入口文件
│   └── pages/                    # 多页面应用
│       ├── match.py              # 匹配测试页
│       └── data_import.py        # 数据导入页
├── config/                       # 配置文件
│   └── app_config.py             # 应用配置
├── logs/                         # 日志目录（git忽略）
├── docs/                         # 文档
│   ├── product.md                # 产品需求
│   ├── dev.md                    # 开发日志（本文件）
│   └── ...
├── tests/                        # 测试代码
│   ├── test_api/
│   └── test_core/
├── Dockerfile                    # 容器构建
├── docker-compose.yml            # 容器编排
├── start.sh                      # Linux启动脚本
├── start.cmd                     # Windows启动脚本
├── requirements.txt              # Python依赖
└── README.md                     # 项目说明
```

---

## 四、API 接口设计

### 4.1 数据匹配接口 `/match`

**请求**:
```json
{
  "query": "找上海地区做AI大模型训练芯片的中试以上项目",
  "resource_types": ["expert", "project", "patent"],
  "top_k": 10,
  "use_hybrid": true
}
```

**响应**:
```json
{
  "request_id": "req_xxx",
  "query_parsed": {
    "intent_text": "AI大模型训练芯片",
    "filters": {
      "region": ["上海市"],
      "maturity_gte": "中试"
    }
  },
  "results_by_type": { ... },
  "meta": { ... }
}
```

### 4.2 数据导入接口 `/data/insert`

**请求**:
```json
{
  "resource_type": "expert",
  "items": [
    {
      "id": "exp_001",
      "fields": { ... },
      "raw_text_for_embedding": "..."
    }
  ]
}
```

### 4.3 查询解析接口 `/parse-query`（内部调试）

**请求**:
```json
{ "query": "找杭州做生物医药小试阶段的企业" }
```

---

## 五、核心流程

```
用户输入 → Redis缓存检查 → LLM解析查询 → BGE-M3向量化
    → Milvus混合检索 + 条件过滤 → 结果融合与排序 → 缓存 → 返回
```

---

## 六、开发计划

### 阶段一：基础设施搭建（Day 1-2） ✅ 已完成
- [x] 创建项目目录结构
- [x] 配置文件 `config/app_config.py`
- [x] 依赖文件 `requirements.txt`
- [x] 日志系统配置（保留7天，INFO级别）
- [x] 基础工具类（Redis缓存、LLM调用封装）

### 阶段二：核心模块开发（Day 3-5） ✅ 已完成
- [x] Embedding 模块（BGE-M3 dense + sparse）
- [x] LLM 查询解析模块（提示词模板 + 结构化输出）
- [x] Milvus 连接与检索模块（混合检索封装）

#### 第二阶段完成详情

**已创建的核心模块：**

| 文件 | 功能 |
|------|------|
| `api/core/embedding.py` | BGE-M3向量生成，支持dense + sparse，带缓存 |
| `api/core/query_parser.py` | LLM查询解析，提取意图和结构化条件 |
| `api/core/milvus_client.py` | Milvus混合检索，支持条件过滤 |
| `api/core/matcher.py` | 统一匹配器，整合所有模块 |

**核心功能说明：**

1. **Embedding模块** (`g_embedding`)
   - 通过One-API调用BGE-M3模型
   - 支持获取dense向量(1024维)和sparse向量
   - 内置Redis缓存（7天有效期）

2. **Query Parser模块** (`g_query_parser`)
   - 使用LLM从自然语言提取意图和条件
   - 输出：intent_text、filters(region/maturity_gte/resource_types)
   - 内置Redis缓存（24小时）

3. **Milvus模块** (`g_milvus`)
   - 自动创建Collection和HNSW/SPARSE索引
   - 混合检索：dense + sparse 加权融合(alpha=0.7, beta=0.3)
   - 支持条件过滤（地域、成熟度向上兼容）

4. **Matcher模块** (`g_matcher`)
   - 统一入口，串联查询解析→向量化→检索
   - 支持多资源类型并行搜索
   - 结果缓存（1小时）

### 阶段三：API开发（Day 6-7） ✅ 已完成
- [x] FastAPI 应用框架
- [x] `/match` 接口实现
- [x] `/data/insert` 接口实现
- [x] `/parse-query` 接口实现

#### 第三阶段完成详情

**已创建的API文件：**

| 文件 | 功能 |
|------|------|
| `api/main.py` | FastAPI主应用，CORS、日志、异常处理 |
| `api/models/schemas.py` | Pydantic数据模型（请求/响应） |
| `api/routers/match.py` | `/api/v1/match` - 数据匹配接口 |
| `api/routers/data.py` | `/api/v1/data/insert` - 数据导入接口 |
| `api/routers/parse.py` | `/api/v1/parse-query` - 查询解析接口 |

**API接口说明：**

1. **POST /api/v1/match**
   - 功能：根据查询文本进行语义匹配
   - 请求：query, resource_types, top_k, use_hybrid
   - 响应：results_by_type（按资源类型分组）

2. **POST /api/v1/data/insert**
   - 功能：批量导入数据到Milvus
   - 请求：resource_type, items（含id/fields/raw_text）
   - 响应：ingested_count, failed_ids

3. **POST /api/v1/parse-query**
   - 功能：调试查询解析效果
   - 请求：query
   - 响应：intent_text, filters

**运行方式：**
```bash
cd f:\project\search_solution
python -m api.main
```

访问地址：http://localhost:5310
API文档：http://localhost:5310/docs

### 阶段四：WebUI开发（Day 8-9） ✅ 已完成
- [x] Streamlit 应用框架
- [x] 匹配测试页面
- [x] 数据导入页面

#### 第四阶段完成详情

**已创建的WebUI文件：**

| 文件 | 功能 |
|------|------|
| `webui/main.py` | Streamlit主应用（包含所有页面） |

**WebUI功能：**

1. **匹配测试页面**
   - 输入查询文本
   - 选择资源类型
   - 配置返回数量和混合检索
   - 展示解析结果和匹配结果

2. **数据导入页面**
   - 支持手动输入单条数据
   - 支持JSON文件批量导入
   - 配置资源类型和字段

3. **关于页面**
   - 系统介绍
   - API服务状态检查

**运行方式：**
```bash
cd f:\project\search_solution
streamlit run webui/main.py --server.port 5320
```

访问地址：http://localhost:5320

### 阶段五：容器化与部署（Day 10） ✅ 已完成
- [x] Dockerfile 编写
- [x] docker-compose.yml 编写
- [x] start.sh / start.cmd 脚本
- [x] README.md 项目说明

#### 第五阶段完成详情

**已创建的部署文件：**

| 文件 | 功能 |
|------|------|
| `Dockerfile` | Docker镜像构建 |
| `docker-compose.yml` | 容器编排（API + WebUI） |
| `start.sh` | Linux启动脚本 |
| `start.cmd` | Windows启动脚本 |
| `README.md` | 项目说明文档 |

**启动方式：**

Windows环境：
```bash
start.cmd api      # 启动API
start.cmd webui    # 启动WebUI
start.cmd all      # 同时启动
```

Linux/Mac环境：
```bash
./start.sh api     # 启动API
./start.sh webui   # 启动WebUI
./start.sh all     # 同时启动
```

Docker部署：
```bash
docker-compose build
docker-compose up -d
```

---

## 七、关键技术点

### 7.1 LLM提示词模板
```
你是一个科技领域信息抽取专家。请从用户输入中提取：
1. 核心技术关键词（用于语义匹配）
2. 结构化过滤条件（地域、成熟度、资源类型等）

输出严格为 JSON，格式如下：
{
  "intent_text": "字符串",
  "filters": {
    "region": ["上海市","杭州市"],
    "maturity_gte": "研发|小试|中试|量产",
    "resource_types": ["expert","project",...]
  }
}

用户输入：{user_query}
```

### 7.2 Milvus Collection 示例（expert）
| 字段名 | 类型 | 说明 |
|-------|------|------|
| id | VARCHAR(64) | 主键 |
| dense_vector | FLOAT_VECTOR(1024) | BGE-M3稠密向量 |
| sparse_vector | SPARSE_FLOAT_VECTOR | BGE-M3稀疏向量 |
| source_id | VARCHAR(64) | 关联元数据ID |
| region | VARCHAR(32) | 标准化地域 |
| maturity | VARCHAR(16) | 成熟度 |
| metadata | JSON | 其他字段快照 |

### 7.3 Redis缓存策略
| Key | TTL | 内容 |
|-----|-----|------|
| `llm_parse:{hash}` | 24h | LLM解析结果 |
| `embedding:{hash}` | 7d | dense + sparse向量 |
| `match_result:{hash}` | 1h | 完整匹配结果 |

### 7.4 CSV数据导入工具

**工具路径：** `tools/import_csv.py`

**功能：** 从CSV文件批量导入数据到向量数据库

**CSV格式要求：**
- 首行为字段名
- 必须包含字段：`id`, `title`, `body`

**使用方法：**

```bash
# 基本用法
python tools/import_csv.py --csv <CSV文件路径> --resource <资源类型>

# 导入tec成果数据
python tools/import_csv.py --csv F:\project\middle_group\tec_data\tec_body.csv --resource tec

# 试运行模式（不实际写入数据，用于测试）
python tools/import_csv.py --csv F:\project\middle_group\tec_data\tec_body.csv --resource tec --dry-run

# 自定义批次大小
python tools/import_csv.py --csv F:\project\middle_group\tec_data\tec_body.csv --resource tec --batch-size 50
```

**参数说明：**

| 参数 | 说明 | 必填 |
|------|------|------|
| `--csv` | CSV文件路径 | 是 |
| `--resource` | 资源类型（expert/project/patent/enterprise/paper/institution/tec） | 是 |
| `--batch-size` | 批次大小，默认100 | 否 |
| `--dry-run` | 试运行模式，不写入数据 | 否 |

**工具特性：**
- 自动读取CSV文件并解析
- 调用BGE-M3生成向量
- 批量写入Milvus向量数据库
- 实时进度条显示
- 详细的成功/失败统计
- 试运行模式支持


```
# 1. 先删除旧的tec Collection
python tools/manage_collection.py --delete ime_tec

# 2. 重新导入数据
python tools/import_csv.py --csv F:\project\middle_group\tec_data\tec_body.csv --resource tec

# 可调整批次大小
python tools/import_csv.py --csv F:\project\middle_group\tec_data\tec_body.csv --resource tec --batch-size 200

python tools/import_csv.py --csv F:\project\middle_group\tec_data\tec_clean.csv --resource tec --batch-size 300

```

---

## 八、配置信息汇总

```python
# One API 配置
ONE_API_KEY = 'sk-GQtwF5ag8p6m8wWf1232B8D5E17f4455A5C14e7a2d393aEe'
ONE_API_BASE_URL = "http://192.168.15.111:3000/v1"

# 模型配置
EMBEDDING_MODEL_NAME = "bge-m3:latest"
LLM_MODEL_NAME = "qwen3"

# Redis
GBL_REDIS_CONFIG = {
    "host": "192.168.15.111",
    "port": 6379,
    "password": "",
    "db": 1
}

# SSDB
SSDB_IP = "192.168.15.111"
SSDB_PORT = 8888

# Milvus
MILVUS_HOST = '192.168.15.111'
MILVUS_PORT = 19530
```

---

## 九、开发进度跟踪

| 日期 | 完成事项 | 备注 |
|------|---------|------|
| 2026-02-09 | 项目规划完成 | 阅读文档，创建开发日志 |
| 2026-02-09 | 第一阶段完成 | 目录结构、配置、日志系统、工具类 |
| 2026-02-09 | 第二阶段完成 | Embedding、LLM解析、Milvus、Matcher模块 |
| 2026-02-09 | 第三阶段完成 | FastAPI框架、3个核心接口 |
| 2026-02-09 | 第四阶段完成 | Streamlit WebUI |
| 2026-02-09 | 第五阶段完成 | Docker容器化、启动脚本 |
| 2026-02-10 | WebUI优化 | 导航改为radio列表样式 |
| 2026-02-10 | 数据导入界面优化 | 增加详细的导入结果反馈和错误提示 |
| 2026-02-10 | 添加得分阈值过滤 | MIN_SCORE_THRESHOLD=0.48，过滤低得分结果 |
| 2026-02-10 | 资源类型result改为tec | 修改所有相关配置和代码 |
| 2026-02-10 | 添加公共title字段 | 所有数据类型支持title字段 |
| 2026-02-10 | 配置文件添加注释 | app_config.py所有参数添加中文注释 |
| 2026-02-10 | 创建CSV导入工具 | tools/import_csv.py 批量导入数据 |

---

## 十、待解决问题

1. 匹配场景.txt 文件编码问题（乱码），需确认原始内容
2. 需准备测试数据集（成果、专家、专利等）
3. 需设计评价体系与评测数据集

---

## 附录：资源类型定义

| 类型 | 说明 | 主要字段 |
|------|------|---------|
| expert | 专家 | 研究方向、职称、单位、地域、专利、论文 |
| project | 项目 | 名称、简介、阶段、地域、专利情况 |
| patent | 专利 | 标题、摘要、申请人、IPC分类 |
| enterprise | 企业 | 名称、产品、专利、投资意向 |
| paper | 论文 | 标题、摘要、作者、期刊 |
| institution | 高校 | 名称、研究领域、地域 |
| tec | 成果 | 名称、简介、阶段、专利情况 |

---

## 十一、项目完成总结

### 11.1 完成的五个阶段

| 阶段 | 内容 | 状态 |
|------|------|------|
| 阶段一 | 基础设施搭建（目录结构、配置、日志、工具类） | ✅ |
| 阶段二 | 核心模块（Embedding、LLM解析、Milvus、Matcher） | ✅ |
| 阶段三 | API开发（FastAPI + 3个核心接口） | ✅ |
| 阶段四 | WebUI开发（Streamlit） | ✅ |
| 阶段五 | 容器化与部署（Docker + 启动脚本） | ✅ |

### 11.2 项目文件结构

```
search_solution/
├── api/
│   ├── main.py              # FastAPI主应用
│   ├── core/
│   │   ├── embedding.py     # BGE-M3向量生成
│   │   ├── query_parser.py  # LLM查询解析
│   │   ├── milvus_client.py # Milvus混合检索
│   │   └── matcher.py       # 统一匹配器
│   ├── routers/
│   │   ├── match.py         # /match接口
│   │   ├── data.py          # /data/insert接口
│   │   └── parse.py         # /parse-query接口
│   ├── models/
│   │   └── schemas.py       # Pydantic数据模型
│   └── utils/
│       ├── logger.py        # 日志系统
│       ├── cache.py         # Redis缓存
│       └── llm_client.py    # LLM客户端
├── webui/
│   └── main.py              # Streamlit WebUI
├── config/
│   └── app_config.py        # 应用配置
├── docs/
│   ├── dev.md               # 开发日志（本文件）
│   └── product.md           # 产品需求
├── logs/                    # 日志目录
├── Dockerfile               # Docker构建
├── docker-compose.yml       # 容器编排
├── start.sh                 # Linux启动脚本
├── start.cmd                # Windows启动脚本
├── requirements.txt         # Python依赖
└── README.md                # 项目说明
```

### 11.3 核心模块功能

| 模块 | 文件 | 功能说明 |
|------|------|---------|
| Embedding | `api/core/embedding.py` | BGE-M3向量生成，支持dense+sparse，带缓存 |
| Query Parser | `api/core/query_parser.py` | LLM查询解析，提取意图和结构化条件 |
| Milvus Client | `api/core/milvus_client.py` | Milvus混合检索，支持条件过滤 |
| Matcher | `api/core/matcher.py` | 统一匹配器，串联查询解析→向量化→检索 |

### 11.4 API接口列表

| 接口 | 方法 | 功能 |
|------|------|------|
| `/api/v1/match` | POST | 根据查询文本进行语义匹配 |
| `/api/v1/data/insert` | POST | 批量导入数据到Milvus |
| `/api/v1/parse-query` | POST | 调试查询解析效果 |
| `/health` | GET | 健康检查 |

### 11.5 快速启动指南

**Windows环境：**
```bash
# 安装依赖
pip install -r requirements.txt

# 启动API服务
start.cmd api

# 启动WebUI（另一个终端）
start.cmd webui

# 或同时启动
start.cmd all
```

**Linux/Mac环境：**
```bash
# 安装依赖
pip install -r requirements.txt

# 启动API服务
./start.sh api

# 启动WebUI（另一个终端）
./start.sh webui

# 或同时启动
./start.sh all
```

**Docker部署：**
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

### 11.6 访问地址

- **API服务**: http://localhost:5310
- **API文档**: http://localhost:5310/docs
- **WebUI**: http://localhost:5320

### 11.7 下一步建议

1. **安装依赖并测试** - 运行 `pip install -r requirements.txt`
2. **准备测试数据集** - 导入一些专家/项目/专利数据
3. **设计评价体系** - 构建评测数据集用于验证匹配效果
4. **性能优化** - 根据实际使用情况调整缓存策略和检索参数
5. **添加单元测试** - 为核心模块编写测试用例
6. **完善错误处理** - 增强异常捕获和用户提示
7. **添加监控指标** - 集成Prometheus/Grafana监控

### 11.8 技术栈汇总

| 类别 | 技术选型 | 版本 |
|------|---------|------|
| 开发语言 | Python | 3.11+ |
| API框架 | FastAPI | 0.109.0 |
| WebUI | Streamlit | 1.30.0 |
| 向量数据库 | Milvus | 2.5.0 |
| 缓存 | Redis | 5.0.1+ |
| LLM | Qwen3 | 通过One-API |
| Embedding | BGE-M3 | 通过One-API |
| 日志 | Loguru | 0.7.2 |
| 容器化 | Docker | - |

---

**项目开发完成时间**: 2026-02-09  
**开发周期**: 1天  
**总代码文件**: 20+ 个
