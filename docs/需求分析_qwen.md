
# 任务：
项目中需要构建一个通用的模块来满足各类匹配场景，详细内容见文档，
请帮我分析这些场景，并对产品进行规划；
# 输出：
输出用户需求的分析；
输出对模块的规划；
列出产品涉及到的技术与支撑；
过程中不需要输出具体的代码；

根据你提供的《匹配场景.txt》文档内容，我们可以将其中描述的多个匹配场景归纳为三大类核心需求。
以下从**用户需求分析**、**模块产品规划**、以及**关键技术与支撑体系**三个方面进行系统性梳理和设计建议。

---

## 一、用户需求分析

### 1. **语义层面的精准匹配需求**
- 用户不再满足于基于关键词字面匹配的结果（如“AI提取关键词+联想词”），而是希望实现**语义相似度匹配**。
- 典型场景包括：
  - 成果/专利/项目之间的相似性判断（用于成果评价、概念验证、科创研判等）；
  - 需求文本与多类型资源（成果、专家、项目、专利、企业、论文、高校）的智能匹配；
  - 关键词驱动的图谱检索（产业/技术/专家图谱）需具备语义泛化能力。

> **核心痛点**：传统关键词匹配无法捕捉技术内涵、上下文语境及隐含关联，导致召回率低、相关性差。

### 2. **多维度约束下的资源筛选需求**
- 用户在匹配时通常附加多种结构化条件，例如：
  - 地域（可向上兼容至省级）；
  - 成熟度阶段（研发 < 小试 < 中试 < 量产，支持向上兼容）；
  - 职称/单位层级（正高 > 副高 > 中级 > 初级，支持向上兼容）；
  - 专利类型、是否拥有专利、投资意向区域等。
- 这些条件需与语义匹配结果**融合过滤与排序**，而非简单交集。

> **核心诉求**：既要“找得准”（语义相关），也要“筛得对”（符合业务规则）。

### 3. **异构资源统一接入与加权融合需求**
- 待匹配对象涵盖7类异构数据源：成果、专利、项目、专家、企业、论文、高校。
- 每类资源有其专属字段与权重策略（如企业匹配中“产品词+50分，专利+30分”）。
- 需要一个**统一接口**接收查询请求，并能按资源类型动态调用不同匹配逻辑与打分模型。

> **挑战**：如何在保持灵活性的同时避免系统碎片化？

---

## 二、模块产品规划

为满足上述需求，建议构建一个名为 **“智能匹配引擎（Intelligent Matching Engine, IME）”** 的通用模块，其架构可分为三层：

### 1. **输入层（Query Ingestion Layer）**
- 支持两类输入：
  - **自由文本查询**（如需求描述、成果简介）；
  - **结构化查询 + 关键词**（如“关键词=量子计算，地域=北京，阶段=中试以上”）。
- 自动识别查询意图（是找相似？还是供需匹配？还是图谱检索？）。

### 2. **处理层（Matching Core Layer）**
该层为核心，包含三个子模块：

#### (1) **语义理解与向量化模块**
- 对输入文本进行语义编码（使用领域微调的 Embedding 模型，如 BGE、E5 或自研科技领域模型）；
- 对库中文本（成果摘要、专利摘要、专家简介等）预生成向量并建立索引（如 FAISS、Milvus）；
- 支持跨模态对齐（如将“需求文本”与“专利摘要”映射到同一语义空间）。

#### (2) **多维约束过滤器（Constraint Filter）**
- 解析用户输入的结构化条件（地域、阶段、职称等）；
- 实现“向上兼容”逻辑（如选择“研发”，则包含小试/中试/量产）；
- 在向量召回后或前进行高效过滤（建议采用“先粗筛后精排”策略）。

#### (3) **资源适配与加权融合模块**
- 针对7类资源定义**匹配配置模板**（YAML/JSON 可配置）：
  - 匹配字段（如专家：研究方向+简介+论文+专利）；
  - 权重规则（如专利命中+30，论文命中+30）；
  - 排序策略（综合得分 = 语义相似度 × α + 结构化匹配分 × β）。
- 支持动态加载新资源类型，无需硬编码。

### 3. **输出层（Result Aggregation & Ranking）**
- 按资源类型分组返回结果；
- 提供统一评分、匹配理由（可解释性）；
- 支持分页、排序、高亮关键匹配字段。

---

## 三、关键技术与支撑体系

| 技术类别 | 具体技术/工具 | 说明 |
|--------|-------------|------|
| **自然语言处理（NLP）** | 领域微调的语义嵌入模型（如 BGE-zh, E5-mistral, 或自研科技大模型） | 实现跨文本的语义相似度计算 |
| **向量数据库** | Milvus / FAISS / PgVector | 高效存储与检索高维向量，支持 ANN 搜索 |
| **规则引擎** | Drools / 自定义 DSL / JSON Schema | 实现“向上兼容”等业务规则的灵活配置 |
| **元数据管理** | 统一资源注册中心（含字段定义、权重模板） | 支撑多类型资源的动态接入 |
| **可解释性（XAI）** | Attention 可视化 / 匹配片段高亮 / 得分分解 | 提升用户信任度 |
| **系统架构** | 微服务 + 异步任务队列（如 Celery / Kafka） | 支持高并发、异步向量化与批量匹配 |
| **评估体系** | 离线 A/B 测试 + 在线点击率/转化率监控 | 持续优化匹配效果 |

---

## 总结

该通用匹配模块的核心价值在于：  
✅ **从“关键词匹配”升级为“语义+规则”双驱动匹配**；  
✅ **统一框架下支持多类资源、多场景、多约束的灵活组合**；  
✅ **通过可配置化设计，降低后续扩展成本**。

建议优先落地“需求找资源”和“成果找相似专利”两个高频场景作为 MVP，再逐步扩展至图谱搜索与企业匹配等复杂场景。

-----------------------------------------

基于最新要求，对技术开发方案进行全面优化与细化。重点强化 **混合检索能力**、**LLM驱动的查询理解**、**baselibs 组件深度集成**，并明确定义 **API 接口规范**。以下是改进后的完整技术方案。

---

## 一、核心能力升级要点

### 1. 启用 **混合检索（Hybrid Search）**
- 利用 **BGE-M3 同时输出稠密向量（dense）与稀疏向量（sparse）**；
- Milvus 2.5.0 原生支持 `SparseVector` 类型，可构建 **稠密 + 稀疏 融合检索**；
- 检索得分 = α × dense_score + β × sparse_score（α, β 可配置）；
- 优势：稠密捕捉语义泛化，稀疏保留关键词精确匹配，兼顾召回率与准确率。

### 2. 用户查询 → LLM 结构化解析
- 用户输入自由文本（如“找北京做量子芯片中试以上阶段的团队”）；
- 通过 **LLM + 提示词工程** 自动提取：
  - 核心意图关键词（用于向量化）
  - 结构化约束条件（地域、成熟度、资源类型等）
- 使用 `baselibs.LLM_TASK` 封装提示词模板与调用逻辑；
- 输出标准化 JSON，供后续过滤与检索使用。

### 3. baselibs 库关键组件使用
| 组件 | 用途 |
|------|------|
| `RedisCache` | 缓存：– LLM 解析结果– 向量– 最终匹配结果 |
| `LLM_TASK` | 封装 Qwen3 调用，加载预设提示词模板，执行结构化抽取 |

---

## 二、系统流程优化（含 LLM 与混合检索）

```mermaid
graph TD
    A[用户输入自由文本] --> B(FastAPI /match)
    B --> C{RedisCache: query_hash?}
    C -- 命中 --> D[返回缓存结果]
    C -- 未命中 --> E[调用 LLM_TASK 解析查询]
    E --> F[输出: {intent_text, filters}]
    F --> G[调用 BGE-M3 获取 dense + sparse 向量]
    G --> H[Milvus Hybrid Search + expr 过滤]
    H --> I[后处理：加权融合 + 资源适配]
    I --> J[生成带解释的匹配结果]
    J --> K[写入 RedisCache]
    K --> L[返回响应]
```

> ✅ 所有 AI 调用（LLM + Embedding）均通过 One-API + OpenAI SDK，统一入口。

---

## 三、API 接口详细设计

### 1. **数据匹配接口** `/match`

#### 请求方法
`POST`

#### 请求头
```http
Content-Type: application/json
Authorization: Bearer <optional_token>
```

#### 请求体（JSON）
```json
{
  "query": "找上海地区做AI大模型训练芯片的中试以上项目",
  "resource_types": ["expert", "project", "patent"],  // 可选，默认全部
  "top_k": 10,                                      // 默认 10
  "use_hybrid": true                                // 是否启用混合检索，默认 true
}
```

#### 响应体（成功，HTTP 200）
```json
{
  "request_id": "req_abc123",
  "query_parsed": {
    "intent_text": "AI大模型训练芯片",
    "filters": {
      "region": ["上海市"],
      "maturity_gte": "中试"
    }
  },
  "results_by_type": {
    "expert": [
      {
        "source_id": "exp_789",
        "name": "张伟",
        "affiliation": "复旦大学",
        "score": 0.92,
        "match_reason": "研究方向包含'AI芯片'，所在单位位于上海，拥有相关专利",
        "highlight_snippet": "...专注于<mark>AI大模型训练芯片</mark>架构设计..."
      }
    ],
    "project": [ /* ... */ ]
  },
  "meta": {
    "total_candidates": 156,
    "filtered_count": 42,
    "retrieval_time_ms": 180
  }
}
```

#### 错误响应（HTTP 4xx/5xx）
```json
{
  "error": "invalid_query",
  "message": "无法解析查询意图，请提供更明确的描述"
}
```

---

### 2. **增量数据导入接口** `/data/insert`

> 用于成果、专家、专利等资源的单条或批量更新。

#### 请求方法
`POST`

#### 请求体（JSON）
```json
{
  "resource_type": "expert",  // 必填：expert/project/patent/enterprise/paper/institution/result
  "items": [
    {
      "id": "exp_001",
      "fields": {
        "name": "李明",
        "research_direction": "量子计算、超导芯片",
        "region": "北京市海淀区",
        "maturity": "量产",
        "patents": ["CN2023XXXXXX"],
        "papers": ["arXiv:2401.xxxx"],
        "raw_text_for_embedding": "李明教授长期从事量子计算与超导芯片研究..."
      }
    }
  ]
}
```

#### 处理逻辑
1. 验证 `resource_type` 是否支持；
2. 提取 `raw_text_for_embedding` 字段（若无，则拼接关键字段）；
3. 调用 BGE-M3 获取 **dense + sparse 向量**；
4. 写入 Milvus 对应 Collection（upsert）；
5. 更新 MySQL 元数据表；
6. 清除相关 Redis 缓存（如该专家曾被匹配过）。

#### 响应
```json
{
  "ingested_count": 1,
  "failed_ids": [],
  "task_id": "task_20260209_001"
}
```

> ⚠️ 支持异步处理（可选 `async=true` 参数），返回任务 ID，通过 `/task/{id}` 查询状态。

---

### 3. **（可选）查询解析调试接口** `/parse-query`（内部使用）

#### 用途
调试 LLM 结构化抽取效果。

#### 请求
```json
{ "query": "找杭州做生物医药小试阶段的企业" }
```

#### 响应
```json
{
  "intent_text": "生物医药",
  "filters": {
    "region": ["杭州市"],
    "maturity_gte": "小试",
    "resource_types": ["enterprise"]
  }
}
```

---

## 四、关键技术实现细节

### 1. **LLM 查询解析提示词（由 LLM_TASK 管理）**

```text
你是一个科技领域信息抽取专家。请从用户输入中提取：
1. 核心技术关键词（用于语义匹配）
2. 结构化过滤条件（地域、成熟度、资源类型等）

输出严格为 JSON，格式如下：
{
  "intent_text": "字符串",
  "filters": {
    "region": ["字符串列表，标准化到市级，如'上海市','杭州市'"],
    "maturity_gte": "研发|小试|中试|量产（若未提及则为null）",
    "resource_types": ["expert","project",...]（若未指定则为null）
  }
}

用户输入：{user_query}
```

> 使用 `baselibs.LLM_TASK(prompt_template, model="qwen3")` 执行，结果自动校验 JSON Schema。

### 2. **Milvus Collection 设计（以 expert 为例）**

| 字段名 | 类型 | 说明 |
|-------|------|------|
| `id` | VARCHAR(64) | 主键（如 exp_001） |
| `dense_vector` | FLOAT_VECTOR(1024) | BGE-M3 稠密向量 |
| `sparse_vector` | SPARSE_FLOAT_VECTOR | BGE-M3 稀疏向量（term_id → weight） |
| `source_id` | VARCHAR(64) | 关联 MySQL 主键 |
| `region` | VARCHAR(32) | 标准化地域（用于 expr 过滤） |
| `maturity` | VARCHAR(16) | 成熟度 |
| `metadata` | JSON | 其他字段快照（用于高亮） |

> 混合检索查询示例（Milvus SDK）：
```python
res = collection.hybrid_search(
    dense_query=dense_vec,
    sparse_query=sparse_vec,
    expr="region in ['上海市'] and maturity in ['中试','量产']",
    limit=top_k,
    rerank=RRF()  # 或 WeightedRanker(alpha=0.7, beta=0.3)
)
```

### 3. **缓存策略（RedisCache）**

| 缓存 Key | TTL | 内容 |
|--------|-----|------|
| `llm_parse:{hash(query)}` | 24h | LLM 解析结果 JSON |
| `embedding:{hash(text)}` | 7d | dense + sparse 向量 |
| `match_result:{hash(full_request)}` | 1h | 完整匹配结果 |

> 使用 `baselibs.RedisCache.setex(key, ttl, value)` 统一管理。

---

## 五、部署与监控建议

- **One-API**：配置两个模型路由：
  - `qwen3` → LLM 生成
  - `bge-m3` → Embedding 生成（需 One-API 支持 sparse 输出）
- **Milvus**：启用 WAL 保证写入可靠性；定期 compact 优化性能；
- **FastAPI**：集成 OpenTelemetry，记录 LLM 调用耗时、Milvus 延迟等指标；
- **Streamlit UI**：调用 `/parse-query` 和 `/match`，用于产品演示与效果验证。

---

该方案已全面支持：
✅ LLM 驱动的查询理解  
✅ BGE-M3 混合检索（dense + sparse）  
✅ baselibs 组件深度集成  
✅ 标准化、可扩展的 API 接口体系  

下一步可进入 **提示词模板验证**、**Milvus 混合检索性能压测** 与 **接口契约文档生成** 阶段。


