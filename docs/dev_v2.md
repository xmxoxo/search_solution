# 多级相似度计算模型 - 开发日志

**创建时间**: 2026-02-11
**项目名称**: 智能匹配引擎 v2 升级（Multi-Level Similarity）

---

## 一、升级背景

原有系统使用单一的 BGE-M3 语义向量进行相似度匹配，存在以下局限性：

1. **语义单一性**：仅基于整体语义相似度，无法区分技术领域、方法、应用场景等维度
2. **可解释性差**：无法解释匹配结果的具体原因
3. **精准度不足**：不同维度的重要性无法区分

**解决方案**：实现"多级相似度计算模型"，从五个维度进行相似度计算和加权融合。

---

## 二、技术方案

### 2.1 五维度相似度架构

| 维度 | 权重 | 说明 |
|------|------|------|
| 语义 (semantic) | 30% | 整体语义相似度，使用原始文本向量化 |
| 领域 (domain) | 25% | 技术领域匹配，LLM提取后再向量化 |
| 方法 (method) | 20% | 技术方法/实现手段匹配 |
| 应用 (application) | 15% | 应用场景/用途匹配 |
| 创新 (innovation) | 10% | 创新点/核心贡献匹配 |

### 2.2 技术实现路径

```
原始文本 → LLM特征提取 → 5个维度文本 → BGE-M3向量化 → 5组向量
                                              ↓
查询文本 → LLM特征提取 → 5个维度向量 → 五维度相似度计算 → 加权融合 → 综合得分
```

### 2.3 两阶段检索

1. **第一阶段（粗召回）**：用语义向量进行向量检索，快速召回候选集（如top 200）
2. **第二阶段（精排）**：对候选集进行五维度相似度计算和加权融合，输出最终排序

---

## 三、核心模块开发

### 3.1 多级特征提取器 (feature_extractor.py)

**文件路径**: `api/core/feature_extractor.py`

**功能**：使用 LLM (Qwen3) 从标题和正文中提取五个维度的特征文本，然后用 BGE-M3 向量化。

**核心类**:
```python
class MultiLevelFeatureExtractor:
    """多级特征提取器 - LLM提取 + BGE-M3向量化"""

    async def extract_all_features(self, title: str, body: str) -> Dict:
        """提取所有维度的特征向量"""
        # 1. LLM提取各维度文本
        # 2. BGE-M3向量化各维度
        # 3. 返回包含5个向量和提取文本的字典
```

**LLM提示词模板**（四个维度）：

1. **领域提取**: 提取技术领域、学科分类、研究方向
2. **方法提取**: 提取技术方法、实现手段、工艺流程
3. **应用提取**: 提取应用场景、用途、适用领域
4. **创新提取**: 提取创新点、核心贡献、技术优势

### 3.2 多级相似度计算器 (similarity_calculator.py)

**文件路径**: `api/core/similarity_calculator.py`

**功能**：计算查询与候选项的五维度相似度，并进行加权融合。

**核心方法**:
```python
class MultiLevelSimilarityCalculator:
    """多级相似度计算器"""

    def calculate(self, query_vectors: Dict, target: Dict) -> Dict:
        """计算综合相似度"""
        # 1. 计算各维度余弦相似度
        # 2. 加权融合得到综合得分
        # 3. 生成相似度解释
```

**权重配置**:
```python
self.weights = {
    "semantic": 0.30,
    "domain": 0.25,
    "method": 0.20,
    "application": 0.15,
    "innovation": 0.10
}
```

---

## 四、Milvus Schema 升级

### 4.1 问题与解决方案

**问题**: Milvus限制一个Collection最多只能有4个向量字段，但我们需要5个（semantic, domain, method, application, innovation）。

**解决方案**: 将5个向量分到两个Collection：

| Collection | 向量字段 | 说明 |
|-----------|---------|------|
| **v2a** | semantic, domain, method | 3个向量，用于第一阶段检索 |
| **v2b** | application, innovation | 2个向量，用于第二阶段计算 |

### 4.2 v2a Collection Schema

**文件路径**: `api/core/milvus_client.py`

**v2a字段**:
| 字段名 | 类型 | 说明 |
|-------|------|------|
| semantic_vector | FLOAT_VECTOR(1024) | 语义向量 |
| domain_vector | FLOAT_VECTOR(1024) | 领域向量 |
| method_vector | FLOAT_VECTOR(1024) | 方法向量 |
| extracted_domain | VARCHAR(500) | 提取的领域文本 |
| extracted_method | VARCHAR(500) | 提取的方法文本 |

**v2b字段**:
| 字段名 | 类型 | 说明 |
|-------|------|------|
| application_vector | FLOAT_VECTOR(1024) | 应用向量 |
| innovation_vector | FLOAT_VECTOR(1024) | 创新向量 |
| extracted_application | VARCHAR(500) | 提取的应用文本 |
| extracted_innovation | VARCHAR(500) | 提取的创新文本 |

### 4.3 Collection命名规则

- `ime_{resource_type}_v2a` - 存储semantic/domain/method向量
- `ime_{resource_type}_v2b` - 存储application/innovation向量

### 4.4 新增方法

| 方法 | 功能 |
|------|------|
| `_get_collection_schema_v2a()` | 创建v2a版本的Collection Schema |
| `_get_collection_schema_v2b()` | 创建v2b版本的Collection Schema |
| `get_or_create_collections_v2()` | 同时获取v2a和v2b两个Collection |
| `insert_v2()` | 插入单条v2数据（同时写入两个Collection） |
| `insert_batch_v2()` | 批量插入v2数据（同时写入两个Collection） |
| `search_v2()` | v2版本向量检索（从两个Collection检索并合并） |

### 4.5 数据写入流程

```
数据 → 拆分为data_a和data_b → 分别写入v2a和v2b
```

### 4.6 数据检索流程

```
用语义向量检索v2a → 根据ID查询v2b获取额外向量 → 合并结果
```

---

## 五、检索逻辑升级 (matcher.py)

### 5.1 新增 match_v2 方法

**文件路径**: `api/core/matcher.py`

**流程**:
```
1. 解析查询 → 获取意图文本
2. 提取查询特征 → 5个维度向量
3. 构建过滤条件
4. 对每个资源类型：
   ├── 第一阶段：用语义向量检索候选集
   ├── 第二阶段：五维度相似度计算和精排
   └── 返回排序后的结果
5. 组装响应（包含各维度分数和解释）
```

### 5.2 MatchResult 扩展

**新增字段**:
```python
class MatchResult:
    dimension_scores: Dict[str, float]  # 各维度得分
    explanation: str                     # 相似度解释
    extracted_info: Dict[str, str]      # 候选项的提取信息
```

---

## 六、响应模型升级 (schemas.py)

**文件路径**: `api/models/schemas.py`

### 6.1 新增模型

```python
class DimensionScores(BaseModel):
    semantic: float = 0.0
    domain: float = 0.0
    method: float = 0.0
    application: float = 0.0
    innovation: float = 0.0

class ExtractedInfo(BaseModel):
    domain: str = ""
    method: str = ""
    application: str = ""
    innovation: str = ""

class QueryFeatures(BaseModel):
    extracted_texts: Optional[Dict[str, str]] = None
```

### 6.2 MatchRequest 新增参数

```python
use_multi_level: bool = Field(default=True, description="是否使用多级相似度计算")
```

---

## 七、数据导入工具升级 (import_csv.py)

**文件路径**: `tools/import_csv.py`

### 7.1 新增功能

1. **批量特征提取**: `extract_features_batch()` - 批量提取多维度特征
2. **v2批量处理**: `process_batch_v2()` - 使用多维度特征进行批量导入
3. **命令行参数**: `--multi-level` - 启用多级特征导入

### 7.2 使用方法

```bash
# 原有方式（单向量）
python tools/import_csv.py --csv F:/data/tec_body.csv --resource tec

# 新方式（多维度特征）
python tools/import_csv.py --csv F:/data/tec_body.csv --resource tec --multi-level
python tools/import_csv.py --csv F:\project\middle_group\tec_data\tec_clean.csv --resource tec --multi-level

```

### 7.3 Collection命名

- 原有方式: `ime_{resource_type}`
- 新方式: `ime_{resource_type}_v2a` 和 `ime_{resource_type}_v2b`

---

## 八、配置文件

**文件路径**: `config/app_config.py`

**新增配置项**:
```python
# 是否默认使用多级相似度计算
USE_MULTI_LEVEL = True
```

---

## 九、文件清单

### 9.1 新建文件

| 文件 | 功能 |
|------|------|
| `api/core/feature_extractor.py` | 多级特征提取器（LLM提取 + BGE-M3向量化） |
| `api/core/similarity_calculator.py` | 多级相似度计算器（五维度加权融合） |
| `docs/solution_2.md` | 技术方案文档 |

### 9.2 修改文件

| 文件 | 修改内容 |
|------|---------|
| `api/core/milvus_client.py` | 新增v2 Schema和相关方法 |
| `api/core/matcher.py` | 新增match_v2方法，两阶段检索 |
| `api/models/schemas.py` | 新增维度分数、提取信息模型 |
| `tools/import_csv.py` | 支持多维度特征导入 |
| `config/app_config.py` | 新增USE_MULTI_LEVEL配置 |

---

## 十、API使用说明

### 10.1 匹配请求

**默认启用多级相似度**:
```json
{
  "query": "找AI大模型训练芯片相关成果",
  "resource_types": ["tec"],
  "top_k": 10
}
```

**禁用多级相似度（使用原有方式）**:
```json
{
  "query": "找AI大模型训练芯片相关成果",
  "resource_types": ["tec"],
  "top_k": 10,
  "use_multi_level": false
}
```

### 10.2 响应示例

```json
{
  "query_parsed": { ... },
  "query_features": {
    "extracted_texts": {
      "domain": "人工智能、大模型、芯片设计",
      "method": "模型训练、芯片架构设计",
      "application": "AI计算、数据中心",
      "innovation": "高效能计算架构"
    }
  },
  "results_by_type": {
    "tec": [
      {
        "id": "tec_001",
        "source_id": "tec_001",
        "score": 0.85,
        "dimension_scores": {
          "semantic": 0.82,
          "domain": 0.88,
          "method": 0.85,
          "application": 0.80,
          "innovation": 0.75
        },
        "explanation": "语义相似度0.82，技术领域高度匹配(0.88)，...",
        "extracted_info": {
          "domain": "人工智能芯片",
          "method": "异构计算",
          "application": "大模型训练",
          "innovation": "低功耗架构"
        }
      }
    ]
  }
}
```

---

## 十一、开发进度

| 日期 | 完成事项 |
|------|---------|
| 2026-02-11 | 技术方案设计完成 |
| 2026-02-11 | 创建多级特征提取器 (feature_extractor.py) |
| 2026-02-11 | 创建多级相似度计算器 (similarity_calculator.py) |
| 2026-02-11 | 升级 Milvus Schema (milvus_client.py) |
| 2026-02-11 | 升级检索逻辑 - 两阶段检索 (matcher.py) |
| 2026-02-11 | 升级响应模型 (schemas.py) |
| 2026-02-11 | 升级数据导入工具 (import_csv.py) |

---

## 十二、关于metadata存储位置

**metadata存储在Milvus的JSON字段中**，具体位置：

1. **Milvus Collection Schema**:
   - 字段名: `metadata`
   - 类型: `DataType.JSON`
   - 用途: 存储原始数据的关键字段（如title、body等）

2. **数据导入时**:
   ```python
   metadata={
       'title': item['title'],
       'body': item['body']
   }
   ```

3. **检索结果中**:
   - `output_fields` 包含 `"metadata"`
   - 返回结果中包含完整的metadata字典

---

## 十三、清空数据命令

### 13.1 使用 manage_collection.py

```bash
# 查看所有Collection
python tools/manage_collection.py --list

# 统计数据量
python tools/manage_collection.py --count

# 删除指定Collection
python tools/manage_collection.py --delete ime_tec
python tools/manage_collection.py --delete ime_tec_v2

# 删除所有IME相关Collection
python tools/manage_collection.py --delete-all
```

### 13.2 常用命令

```bash
# 清空所有v1数据
python tools/manage_collection.py --delete ime_tec
python tools/manage_collection.py --delete ime_patent
python tools/manage_collection.py --delete ime_project
# ... 其他资源类型

# 清空所有v2数据
python tools/manage_collection.py --delete ime_tec_v2_a
python tools/manage_collection.py --delete ime_tec_v2_b
python tools/manage_collection.py --delete ime_patent_v2_a
python tools/manage_collection.py --delete ime_patent_v2_b
# ... 其他资源类型

# 一键清空所有
python tools/manage_collection.py --delete-all
```

---

## 十五、WebUI界面升级

### 15.1 界面调整

- 去掉"启用混合检索"选项，默认启用
- 增加"启用多维度匹配"选项，默认勾选
- 资源类型默认选中改为"tec"

### 15.2 多维度匹配结果展示

启用多维度匹配时，结果展示包含：

1. **查询特征提取** - 展示查询文本的五维度提取结果
2. **各维度得分** - 显示语义、领域、方法、应用、创新五个维度的具体分数
3. **匹配解释** - 显示相似度计算的解释说明
4. **提取信息** - 展示候选项的提取特征

---

## 十六、问题修复记录

### 16.1 Milvus向量字段限制问题

**问题**: Milvus限制一个Collection最多只能有4个向量字段

**错误信息**:
```
MilvusException: (code=65535, message=maximum vector field's number should be limited to 4)
```

**解决方案**: 将5个向量分到两个Collection：
- v2a: semantic + domain + method (3个向量)
- v2b: application + innovation (2个向量)

```
# 删除旧的v2 Collection
python tools/manage_collection.py --delete ime_tec_v2_a
python tools/manage_collection.py --delete ime_tec_v2_b

# 重新导入数据
python tools/import_csv.py --csv data/tec_clean.csv --resource tec --multi-level

```

**修改文件**:
- `api/core/milvus_client.py` - 重构Schema和插入/检索逻辑
- `tools/import_csv.py` - 更新collection_name引用

### 16.2 导入性能优化 - 向量调用合并

**问题**: 批量导入数据时，每条数据需要调用5次向量化接口（semantic + 4个维度），导致大量API调用和延迟

**原流程**:
```
for item in batch:
    semantic_vector = embedding(item.text)
    domain_vector = embedding(domain_text)
    method_vector = embedding(method_text)
    application_vector = embedding(application_text)
    innovation_vector = embedding(innovation_text)
```

**优化方案**: 将同一批次内的所有向量化调用合并为批量调用

**新流程**:
```
1. 收集所有文本: [text1, text2, ..., textN, domain1, method1, application1, innovation1, ...]
2. 一次批量调用获取所有向量
3. 按索引分配给对应条目
```

**优化效果**:
- 假设batch_size=10，原流程需要50次向量化调用
- 优化后只需2次批量调用（语义向量 + 维度向量）
- 显著减少API调用次数和网络延迟

**修改文件**:
- `api/core/feature_extractor.py` - 新增 `extract_all_features_batch()` 方法
- `api/core/embedding.py` - 已有 `get_embeddings_batch()` 批量向量化方法
- `tools/import_csv.py` - 使用优化后的批量特征提取方法

### 16.3 API接口修复 - use_multi_level参数传递

**问题**: API接口 `/api/v1/match` 没有传递 `use_multi_level` 参数，导致无法控制匹配模式

**修复**: 在 `api/routers/match.py` 中添加参数传递

```python
result = await g_matcher.match(
    query=body.query,
    resource_types=body.resource_types,
    top_k=body.top_k,
    use_hybrid=body.use_hybrid,
    use_cache=body.use_cache,
    use_multi_level=body.use_multi_level  # 新增
)
```

**修改文件**:
- `api/routers/match.py` - 添加 `use_multi_level` 参数传递，添加 `QueryFeatures` 导入

### 16.4 LLM提示词优化 - 禁止编造内容

**问题**: LLM在提取关键词时会编造内容，如 query="近视防控" 时编造创新点

**优化方案**: 重新设计4个维度的提示词

**新提示词特点**:
1. 明确角色："专业的文本分析助手"
2. 严格规则："只基于用户提供的文本内容进行提取，严禁编造"
3. 空值处理："如果没有明确提到，直接输出'无'"
4. 格式要求：详细规则 + 中文逗号分隔
5. 示例引导：每个维度都添加正反示例

**修改文件**:
- `api/core/feature_extractor.py` - 重写 `DOMAIN_EXTRACT_PROMPT`、`METHOD_EXTRACT_PROMPT`、`APPLICATION_EXTRACT_PROMPT`、`INNOVATION_EXTRACT_PROMPT`

## 十七、v2架构重构 - 复用v1混合检索

### 17.1 方案调整

**问题**: v2版本独立存储语义向量，与v1的dense_vector重复，浪费存储空间

**最终方案**:
- v1的混合检索（dense + sparse）作为第一阶段召回
- v2只存储4个扩展维度向量（domain、method、application、innovation）
- 检索流程：v1混合检索获取候选 → v2多维度精排

**优势**:
- 减少存储空间（v2a不再存储semantic_vector）
- 复用v1的混合检索能力
- v2专注于扩展维度的特征匹配

### 17.2 五维度相似度计算

| 维度 | 数据来源 | 计算方式 |
|------|----------|----------|
| semantic | v1混合检索得分 | 直接使用 |
| domain | v2a.domain_vector | 余弦相似度 |
| method | v2a.method_vector | 余弦相似度 |
| application | v2b.application_vector | 余弦相似度 |
| innovation | v2b.innovation_vector | 余弦相似度 |

**权重配置**（写入 `config/app_config.py`）:
```python
MULTI_LEVEL_WEIGHTS = {
    "semantic": 0.30,
    "domain": 0.25,
    "method": 0.20,
    "application": 0.15,
    "innovation": 0.10
}
```

**特殊处理**:
- 如果某维度的提取文本为空或"无"，该维度得分为0

### 17.3 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `config/app_config.py` | 添加 `MULTI_LEVEL_WEIGHTS` 配置 |
| `api/core/milvus_client.py` | v2a移除semantic_vector，search_v2改用v1混合检索 |
| `api/core/similarity_calculator.py` | 使用配置权重，支持semantic_score参数 |
| `api/core/matcher.py` | match_v2支持use_hybrid参数，调用流程调整 |
| `tools/import_csv.py` | v2导入移除semantic_vector |
| `tools/migrate_v1_to_v2.py` | 新增数据迁移工具 |

### 17.4 数据迁移工具

**功能**: 从v1 collection迁移数据到v2 collections

**使用方式**:
```bash
# 模拟迁移（不实际写入）
python tools/migrate_v1_to_v2.py --resource tec --dry-run

# 实际迁移
python tools/migrate_v1_to_v2.py --resource tec
```


```
# 默认参数（推荐）
python tools/migrate_v1_to_v2.py --resource tec

# 更小的批次，更慢但更稳定
python tools/migrate_v1_to_v2.py --resource tec --batch-size 5

# 断点续传（从第1000条开始）
python tools/migrate_v1_to_v2.py --resource tec --start-index 1000
```

**工作流程**:
1. 从v1 collection读取所有数据
2. 对每条数据提取4个维度的LLM特征
3. 批量插入到v2a和v2b collections

## 十八、LLM调用优化 - 合并4次调用为1次

### 18.1 问题分析

**原方案**: 每个维度调用一次LLM，共4次调用
```
domain → LLM → domain_text
method → LLM → method_text
application → LLM → application_text
innovation → LLM → innovation_text
```

**问题**:
1. 4次API调用，增加延迟和成本
2. 容易触发API限流（401错误）
3. 代码复杂度高

### 18.2 优化方案

**新方案**: 合并为1次调用，JSON格式输出
```
统一提示词 → LLM → {"domain":"...","method":"...","application":"...","innovation":"..."}
```

**新提示词** (`MULTI_DIMENSION_EXTRACT_PROMPT`):
- 一次分析4个维度：domain、method、application、innovation
- 严格禁止编造内容
- 无内容时输出"无"
- JSON格式输出

### 18.3 短文本处理

对于"近视防控"这类短文本，4个维度的提取结果都为"无"是合理的。

**处理逻辑**:
- 如果4个扩展维度都为"无"，则只使用semantic维度进行匹配
- 其他4个维度得分为0，不影响最终结果
- 自动降级为混合检索模式


已创建诊断工具 tools/test_llm_connection.py 
运行方式

```
python tools/test_llm_connection.py
```

####  工具功能
这个工具会依次测试：

1. LLM连接 - 测试与One-API的LLM服务连接
2. Embedding连接 - 测试向量嵌入服务
3. FeatureExtractor - 测试完整的特征提取流程
#### 可能的问题原因
根据错误信息 令牌验证失败 ，可能的原因：

1. API Key格式问题 - 检查是否有空格或特殊字符
2. 并发限制 - 数据迁移时大量并发请求可能触发限流
3. 请求频率 - 短时间内请求过多
运行诊断工具后，根据结果再决定如何处理。如果是限流问题，可能需要在数据迁移工具中添加延迟或降低并发。


### 18.4 修改文件

| 文件 | 修改内容 |
|------|----------|
| `api/core/feature_extractor.py` | 合并4个提示词为1个，使用`g_llm.parse_json()` |
| `api/utils/llm_client.py` | 提供`parse_json()`方法支持JSON输出 |

### 18.5 优化效果

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| LLM调用次数 | 4次/条 | 1次/条 |
| API调用量 | 减少75% | - |
| 触发限流风险 | 高 | 低 |
| 代码复杂度 | 4个提示词维护 | 1个提示词 |

## 十九、自动降级机制 - 短文本处理

### 19.1 问题

当query的4个维度`extracted_texts`全为"无"或空时，多维度匹配得分会很低，导致没有结果能超过阈值。

**示例**: query="近视防控"
```json
"extracted_texts": {
    "domain": "无",
    "method": "无",
    "application": "无",
    "innovation": "无"
}
```

### 19.2 解决方案

**自动降级机制**: 当query的4个维度都为空时，自动降级为v1混合检索模式。

**判断逻辑**:
```python
all_empty = all(
    extracted_texts.get(dim, "无") in ("无", "", None)
    for dim in ["domain", "method", "application", "innovation"]
)

if all_empty:
    # 降级为v1混合检索
    return await self._match_v1_internal(...)
```

### 19.3 降级流程

```
query → 提取4个维度特征
         ↓
    检测是否全为"无"
         ↓
    ┌────┴────┐
    ↓         ↓
  全为空    有内容
    ↓         ↓
 降级v1    正常v2匹配
```

### 19.4 降级标识

降级后的返回结果中会添加标识：
```json
"query_features": {
    "extracted_texts": {"domain": "无", "method": "无", "application": "无", "innovation": "无"},
    "fallback_to_v1": true
}
```

### 19.5 修改文件

| 文件 | 修改内容 |
|------|----------|
| `api/core/matcher.py` | 添加自动降级检测和`_match_v1_internal`方法 |

## 二十、性能优化 - 空值跳过向量化

### 20.1 问题

对于"近视防控"这类短文本，4个维度的提取结果都为"无"，但仍然会调用4次向量化接口，造成浪费。

### 20.2 优化方案

**跳过空值向量化**：
- 如果某维度提取结果为"无"或空，跳过向量化
- 空维度使用 `EMPTY_VECTOR = []`
- 在相似度计算时，空向量直接得0分

**代码示例** (`feature_extractor.py`):
```python
def is_empty_value(value: str) -> bool:
    """判断值是否为空"""
    return not value or value.strip() == "" or value.strip() == "无"

# 只向量化非空的维度
for dim in DIMENSIONS:
    text = extracted_texts.get(dim, "")
    if not is_empty_value(text):
        embed_indices[dim] = len(texts_to_embed)
        texts_to_embed.append(text)
```

### 20.3 query空维度处理

在 `similarity_calculator.calculate()` 中：
- 如果query的某维度为空，该维度直接得0
- 不需要计算余弦相似度

```python
query_empty = is_empty_value(query_extracted_texts.get(dim, ""))
if query_empty or target_empty or not query_vec or not target_vec:
    scores[dim] = 0.0
```

### 20.4 修改文件

| 文件 | 修改内容 |
|------|----------|
| `api/core/feature_extractor.py` | 跳过空值向量化，添加`is_empty_value`函数 |
| `api/core/similarity_calculator.py` | 支持query空维度判断，添加`query_extracted_texts`参数 |
| `api/core/matcher.py` | 传递`query_extracted_texts`给相似度计算器 |

### 20.5 优化效果

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| 4个维度全空 | 4次向量化 | 0次向量化 |
| 2个维度有值 | 4次向量化 | 2次向量化 |
| 平均情况 | 4次/条 | **1-2次/条** |

## 二十一、数据迁移工具优化 - 批量并行处理

### 21.1 问题

原迁移工具速度慢，原因：
1. 逐条处理LLM和向量化
2. 批次太小（10条/批）
3. 没有并行处理

### 21.2 优化方案

**1. 使用批量特征提取**：
```python
# 优化前：逐条处理
for item in batch:
    features = await extract_features(item)

# 优化后：批量处理
features_list = await g_feature_extractor.extract_all_features_batch(items)
```

**2. 增大批次**：
- BATCH_SIZE: 10 → 50

**3. 并行处理**：
- MAX_PARALLEL_BATCHES: 3批次并行
```python
for i in range(0, len(batches), MAX_PARALLEL_BATCHES):
    parallel_batches = batches[i:i+MAX_PARALLEL_BATCHES]
    results = await asyncio.gather(*tasks)
```

### 21.3 新参数

```bash
python tools/migrate_v1_to_v2.py \
    --resource tec \
    --batch-size 50 \
    --parallel 3
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 50 | 每批处理条数 |
| `--parallel` | 3 | 同时处理的批次数 |

### 21.4 性能对比

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 批次大小 | 10 | 50 | 5x |
| 并行度 | 1 | 3 | 3x |
| 向量化调用 | 4次/条 | 1-2次/条 | 2-4x |
| **预估总时间** | 16小时 | **1-2小时** | **10-15x** |

### 21.5 修改文件

| 文件 | 修改内容 |
|------|----------|
| `tools/migrate_v1_to_v2.py` | 批量特征提取、并行处理、增大批次 |

## 二十二、权重动态归一化

### 22.1 问题

原权重计算方式：所有维度都参与计算，即使得分为0。

**问题**：当某些维度得分为0时，总得分会被稀释。

### 22.2 优化方案

**动态权重归一化**：只计算非零维度的权重和。

```python
weight_array = np.array([self.weights.get(dim, 0) for dim in self.dimensions])
score_array = np.array([scores.get(dim, 0) for dim in self.dimensions])

nonzero_mask = score_array != 0
total_weight = np.sum(weight_array[nonzero_mask])

if total_weight == 0:
    total_score = 0.0
else:
    total_score = np.sum(weight_array * score_array) / total_weight
```

### 22.3 效果

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| 5个维度都有值 | 正常 | 正常 |
| 只有2个维度有值 | 得分被稀释 | 只用2个维度计算 |
| 只有1个维度有值 | 得分被严重稀释 | 只用1个维度计算 |

### 22.4 修改文件

| 文件 | 修改内容 |
|------|----------|
| `api/core/similarity_calculator.py` | 实现动态权重归一化 |

## 二十三、数据预处理工具

### 23.1 问题

数据导入时，LLM特征提取是最耗时的操作。

**耗时分析**：
- LLM调用：1-3秒/条
- 向量化：0.4秒/条
- Milvus插入：0.1秒/条

### 23.2 优化方案

**预处理流程**：
1. 使用 `preprocess_csv.py` 预处理CSV文件
2. 批量提取4个维度的特征文本
3. 生成新的CSV文件（包含提取的特征文本）
4. 导入时使用 `--preprocessed` 参数，跳过LLM

### 23.3 预处理工具

**工具**：`tools/preprocess_csv.py`

**功能**：
- 批量LLM特征提取
- 生成包含4个维度特征文本的CSV
- 支持自定义批次大小

**使用方式**：
```bash

python tools/preprocess_csv.py \
    --input data/tec_clean.csv \
    --output data/tec_body_preprocessed.csv \
    --batch-size 50
```

**输出格式**：
```csv
id,source_id,title,body,region,maturity,metadata,extracted_domain,extracted_method,extracted_application,extracted_innovation
```

### 23.4 导入预处理文件

**原方式**：
```bash
python tools/import_csv.py --csv data/tec_clean.csv --resource tec --multi-level
```

**新方式**：
```bash
# 1. 预处理
python tools/preprocess_csv.py --input data/tec_clean.csv --output data/tec_body_preprocessed.csv

# 2. 导入（跳过LLM）
python tools/import_csv.py \
    --csv data/tec_body_preprocessed.csv \
    --resource tec \
    --multi-level \
    --preprocessed
```

### 23.5 性能对比

| 操作 | 原方式 | 预处理方式 |
|------|--------|----------|
| LLM调用 | 导入时实时 | 预处理一次性 |
| 向量化 | 导入时实时 | 导入时批量 |
| **总耗时** | 1.5-3.5秒/条 | **0.5-1秒/条** |
| **3万条数据** | 16小时 | **5-8小时** |

### 23.6 修改文件

| 文件 | 修改内容 |
|------|----------|
| `tools/preprocess_csv.py` | 新增数据预处理工具 |
| `tools/import_csv.py` | 添加 `--preprocessed` 参数和 `process_batch_v2_preprocessed` 方法 |

## 二十四、API接口优化 - 去掉use_hybrid参数

### 24.1 修改内容

**1. 去掉 `use_hybrid` 参数**
- `MatchRequest` 中移除 `use_hybrid` 字段
- 混合检索默认始终启用（`use_hybrid=True`）
- 简化API调用，减少用户配置

**2. v1结果作为保底数据**
- 在 `use_multi_level=true` 模式下
- v1混合检索结果作为保底数据
- 与v2多维度组合匹配结果合并后统一排序
- 确保返回结果数量充足

### 24.2 实现逻辑

**match_v2 方法优化**：
```python
# 1. 获取v2候选结果
candidates = await g_milvus.search_v2(...)

# 2. 获取v1保底结果
v1_results = await g_milvus.hybrid_search(...)

# 3. 合并去重
v1_result_ids = {r["id"] for r in v1_results}

# 4. 添加v1保底结果
for v1_item in v1_results:
    if v1_item["id"] not in v1_result_ids:
        # 添加保底结果，标记为语义匹配
        scored_results.append(MatchResult(
            score=v1_item["score"],
            dimension_scores={"semantic": v1_item["score"], ...},
            explanation="语义匹配（保底）"
        ))

# 5. 统一排序
scored_results.sort(key=lambda x: x.score, reverse=True)
```

### 24.3 效果

| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| v2结果充足 | 只返回v2结果 | 返回v2结果 |
| v2结果不足 | 返回少量结果 | **v2+v1保底，保证数量** |
| 结果排序 | 按v2得分排序 | **统一按得分排序** |

### 24.4 修改文件

| 文件 | 修改内容 |
|------|----------|
| `api/models/schemas.py` | `MatchRequest` 移除 `use_hybrid` 字段 |
| `api/routers/match.py` | 移除 `use_hybrid` 参数传递，meta中固定为True |
| `api/core/matcher.py` | `match_v2` 方法添加v1保底数据合并逻辑 |

## 十四、下一步工作

1. **测试数据导入** - 使用 `--multi-level` 参数导入测试数据
2. **验证检索效果** - 对比v1和v2的检索结果差异
3. **权重调优** - 根据实际效果调整五维度权重
4. **性能优化** - 优化LLM特征提取的批量处理效率
5. **缓存策略** - 为LLM提取结果添加缓存机制

---

## 二十五、MySQL专利数据导入工具

### 25.1 工具功能

**文件路径**: `tools/import_mysql_patent.py`

**功能**: 直接从MySQL数据库导入专利数据到Milvus向量数据库，支持批量处理、错误重试和失败记录追踪。

### 25.2 技术实现

**核心特性**:

1. **数据库连接**：使用 `app_config.py` 中的 `DATABASE_CONNECT_STRING` 变量
2. **批量处理**：默认 `batch_size=10` 条/批
3. **错误处理**：
   - 自动重试机制（最多3次）
   - 失败记录ID保存到文件
   - 各类异常捕获，避免导入中断
4. **存储优化**：
   - 标题(title)和内容(body)字段不保存到Milvus
   - 只存储向量和提取的特征文本
5. **LLM模型**：默认使用 `qwen3:32b` 进行特征提取（通过环境变量设置）
6. **数据过滤**：
   - 过滤 `patent_type`、`patent_name`、`descript` 为空的记录
   - 总数约291万条数据

### 25.3 SQL查询语句

**数据查询**:
```sql
SELECT id, patent_name as title, descript as body FROM t_patent_trade_info
WHERE patent_type IS NOT NULL AND patent_name IS NOT NULL AND descript IS NOT NULL
ORDER BY id LIMIT 100
```

**总数统计**:
```sql
SELECT COUNT(*) FROM t_patent_trade_info
WHERE patent_type IS NOT NULL AND patent_name IS NOT NULL AND descript IS NOT NULL
```

**总数**: 2914891

### 25.4 使用方法

**基本用法**:
```bash
# 默认导入所有专利数据
python tools/import_mysql_patent.py

# 导入指定数量的数据（测试用）
python tools/import_mysql_patent.py --limit 100

# 自定义批次大小
python tools/import_mysql_patent.py --batch-size 20

# 干运行模式（仅统计，不实际导入）
python tools/import_mysql_patent.py --dry-run

# 完整命令示例
python tools/import_mysql_patent.py --resource patent --batch-size 10 --limit 1000
```

**命令行参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--resource` | `patent` | 资源类型 |
| `--batch-size` | `10` | 每批处理条数 |
| `--limit` | `None` | 限制导入数据量 |
| `--dry-run` | `False` | 干运行模式 |
| `--no-save-intermediate` | `False` | 不保存中间数据文件 |
| `--from-file` | `None` | 从中间数据文件导入 |

### 25.5 中间数据保存机制

**功能说明**:
- 导入过程中自动保存中间数据到CSV文件
- 支持断点续传，避免重复提取特征
- 每个文件最多保存10000条数据
- 文件保存在 `data/` 目录下

**文件命名规则**:
```
patent_{begin_id}_{resource_type}.csv
```
例如: `patent_1_patent.csv`, `patent_10001_patent.csv`

**CSV文件格式**:
```csv
id,source_id,region,maturity,domain_vector,method_vector,application_vector,innovation_vector,extracted_domain,extracted_method,extracted_application,extracted_innovation
1,1,,,[0.1,0.2,...],[0.3,0.4,...],[0.5,0.6,...],[0.7,0.8,...],人工智能,深度学习,医疗,创新点
```

**字段说明**:
| 字段 | 说明 |
|------|------|
| id | 专利ID |
| source_id | 源ID（与id相同） |
| region | 地域（空） |
| maturity | 成熟度（空） |
| domain_vector | 领域向量（JSON格式） |
| method_vector | 方法向量（JSON格式） |
| application_vector | 应用向量（JSON格式） |
| innovation_vector | 创新向量（JSON格式） |
| extracted_domain | 提取的领域文本 |
| extracted_method | 提取的方法文本 |
| extracted_application | 提取的应用文本 |
| extracted_innovation | 提取的创新文本 |

**使用场景**:
1. **断点续传**: 导入中断后，可以从中间数据文件继续导入
2. **快速重试**: Milvus导入失败时，直接从CSV文件重试，无需重新提取特征
3. **数据备份**: 保存提取的特征和向量，便于后续分析和调试

**从中间数据文件导入**:
```bash
# 从CSV文件导入到Milvus
python tools/import_mysql_patent.py --from-file data/patent_1_patent.csv --batch-size 50
```

### 25.6 错误处理和日志

**失败记录**:
- 失败的专利ID会保存到 `failed_patent_ids_{resource_type}.txt` 文件
- 格式：每行一个ID，便于后续检查和重新处理

**日志输出**:
- 导入过程中的详细信息会写入日志文件
- 包括：总数据量、成功/失败数量、处理时间等

**重试机制**:
- 遇到网络或服务错误时，自动重试最多3次
- 每次重试间隔递增（5秒、10秒、15秒）

### 25.6 存储策略

**Milvus存储结构**:
- **v1 Collection (ime_patent)**: 
  - dense_vector: BGE-M3 稠密向量（语义向量）
  - sparse_vector: BGE-M3 稀疏向量（关键词向量）
  - source_id: 原始ID
  - region: 地域（空）
  - maturity: 成熟度（空）
  - metadata: 元数据（空，节省空间）
- **v2a Collection (ime_patent_v2a)**:
  - domain_vector: 领域向量
  - method_vector: 方法向量
  - extracted_domain: 提取的领域文本
  - extracted_method: 提取的方法文本
- **v2b Collection (ime_patent_v2b)**:
  - application_vector: 应用向量
  - innovation_vector: 创新向量
  - extracted_application: 提取的应用文本
  - extracted_innovation: 提取的创新文本

**优势**:
- v1 提供混合检索能力（dense + sparse）
- v2 提供多维度特征匹配
- 复用 v1 的语义向量，避免重复存储
- 专注于向量相似度计算 

### 25.7 性能考虑

**处理速度**:
- 每批10条数据，包含LLM特征提取和向量化
- 预计处理速度：约10-15秒/批
- 完整导入291万条数据：预计需要约100-150小时

**优化建议**:
1. 可使用 `--limit` 参数分批导入
2. 考虑在非高峰期运行导入任务
3. 监控系统资源使用情况

### 25.8 指定起始ID

###＃　 从指定 ID 之后导入
```
# 从 指定的ID 之后开始导入 100条
python tools/import_mysql_patent.py --start-id "CN00100544.8" --limit 100

# 接着继续导入100条 
python tools/import_mysql_patent.py --limit 100
```

#### 完整命令示例

```
# 从 ID 10000 之后导入，限制 1000 条，批次大小 50
python tools/import_mysql_patent.py --start-id 10000 --limit 1000 --batch-size 50

# 从 ID 100000 之后导入，不保存中间数据
python tools/import_mysql_patent.py --start-id 100000 --no-save-intermediate
```

每次导入成功后会将最新ID号写入位置文件: "data/last_patent_id.txt"
不指定start-id时，默认从位置文件加载最新的ID；


---

## 二十六、v1+v2双版本导入

### 26.1 问题发现

运行 `import_mysql_patent.py` 后发现只导入了 v2 版本的扩展维度特征，缺少基础的混合检索向量（dense_vector 和 sparse_vector），导致无法进行 v1 混合检索。

### 26.2 解决方案

修改 `import_mysql_patent.py`，同时导入 v1 和 v2 的数据：

**v1 数据（ime_patent Collection）**:
- dense_vector: BGE-M3 稠密向量
- sparse_vector: BGE-M3 稀疏向量
- source_id: 原始ID
- region: 地域（空）
- maturity: 成熟度（空）
- metadata: 元数据（空）

**v2 数据（ime_patent_v2a 和 ime_patent_v2b Collections）**:
- domain_vector: 领域向量
- method_vector: 方法向量
- application_vector: 应用向量
- innovation_vector: 创新向量
- extracted_texts: 提取的文本

### 26.3 代码修改

**修改文件**:
- `api/core/milvus_client.py`: 添加 `insert_batch` 方法用于批量插入 v1 数据
- `tools/import_mysql_patent.py`: 
  - 修改 `process_batch` 函数，同时生成并插入 v1 和 v2 数据
  - 修改 `save_intermediate_data` 函数，保存 v1 和 v2 的向量
  - 修改 `load_intermediate_data` 函数，加载 v1 和 v2 的向量
  - 修改 `import_from_intermediate_file` 函数，同时导入 v1 和 v2 数据

**关键代码**:
```python
# 生成v1混合检索向量
full_text = f"{title} {body}"
dense_vector, sparse_vector = await g_embedding.get_embedding(full_text, use_cache=False)

# 生成v2扩展维度向量
features = await g_feature_extractor.extract_all_features(title, body)

# v1数据（混合检索）
v1_insert_items.append({
    'id': patent['id'],
    'dense_vector': dense_vector,
    'sparse_vector': sparse_vector,
    'source_id': patent['id'],
    'region': '',
    'maturity': '',
    'metadata': {}
})

# v2数据（扩展维度）
v2_insert_items.append({
    'id': patent['id'],
    'vectors': {
        'domain_vector': features['domain_vector'],
        'method_vector': features['method_vector'],
        'application_vector': features['application_vector'],
        'innovation_vector': features['innovation_vector']
    },
    'extracted_texts': features['extracted_texts'],
    'source_id': patent['id'],
    'region': '',
    'maturity': '',
    'metadata': {}
})
```

### 26.4 使用方法

**从 MySQL 导入**:
```bash
# 同时导入 v1 和 v2 数据
python tools/import_mysql_patent.py --resource patent --batch-size 10 --limit 100
```

**从中间文件导入**:
```bash
# 从中间文件同时导入 v1 和 v2 数据
python tools/import_mysql_patent.py --from-file data/patent_1_patent.csv --resource patent
```


```
从当前ID继续导入：
python tools/import_mysql_patent.py 
python tools/import_mysql_patent.py --batch-size 20


从指定的ID 导入 10条
python tools/import_mysql_patent.py --start-id "CN00100544.8" --limit 10
python tools/import_mysql_patent.py --start-id "CN00102970.3" --limit 10



从当前ID 继续导入20条
python tools/import_mysql_patent.py --limit 20

从指定文件导入
python tools/import_mysql_patent.py --from-file data/patent_CN00100160.4_patent.csv --resource patent


python tools/import_mysql_patent.py --start-id "CN200410091839.7"

```

分段导入：


```
[
	2520001,
	"CN202220760856.9"
],
[
	2550001,
	"CN202221673402.4"
],
[
	2580001,
	"CN202222368070.5"
],
[
	2610001,
	"CN202223047465.1"
],
2640001, "CN202230489442.2"

python tools/import_mysql_patent.py --start-id "CN202220760856.9" --limit 30000
python tools/import_mysql_patent.py --start-id "CN202221673402.4" --limit 30000
python tools/import_mysql_patent.py --start-id "CN202222368070.5" --limit 30000
python tools/import_mysql_patent.py --start-id "CN202223047465.1" --limit 30000

python tools/import_mysql_patent.py --start-id "CN202230489442.2"

```


### 26.5 中间数据文件格式

**CSV 文件包含以下字段**:
- id: 原始ID
- source_id: 源ID
- region: 地域
- maturity: 成熟度
- dense_vector: v1 稠密向量（JSON格式）
- sparse_vector: v1 稀疏向量（JSON格式）
- domain_vector: v2 领域向量（JSON格式）
- method_vector: v2 方法向量（JSON格式）
- application_vector: v2 应用向量（JSON格式）
- innovation_vector: v2 创新向量（JSON格式）
- extracted_domain: 提取的领域文本
- extracted_method: 提取的方法文本
- extracted_application: 提取的应用文本
- extracted_innovation: 提取的创新文本

**文件命名规则**: `patent_{begin_id}_{resource_type}.csv`

**文件大小限制**: 每个文件最多保存 10000 条数据

### 26.6 性能影响

**处理时间**:
- 每批10条数据需要生成 6 个向量（v1: 2个，v2: 4个）
- 预计处理速度：约 15-20 秒/批
- 完整导入291万条数据：预计需要约 150-200 小时

**存储空间**:
- v1 Collection: 约 2.9M × 1024 × 4 bytes ≈ 11.9 GB（dense_vector）
- v2a Collection: 约 2.9M × 1024 × 4 bytes × 2 ≈ 23.8 GB
- v2b Collection: 约 2.9M × 1024 × 4 bytes × 2 ≈ 23.8 GB
- 总计约 60 GB（不含索引和元数据）

---

**开发完成时间**: 2026-02-14
