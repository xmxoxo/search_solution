# 多级相似度计算模型 - 技术方案

## 一、方案背景

当前系统使用单一向量检索（dense + sparse 混合检索），基于余弦相似度进行简单排序，缺乏多维度特征和权重配置。

本方案通过 **LLM提取 + BGE-M3向量化** 的方式，实现五维度相似度计算，每个维度独立计算后加权融合。

---

## 二、核心架构

### 2.1 整体流程

```
┌────────────────────────────────────────────────────────────────────┐
│                         特征提取阶段                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │  原始文本   │ →  │  LLM 提取   │ →  │  5个维度的关键文本      │ │
│  │  (title+    │    │  (Qwen3)    │    │  (domain/method/...)    │ │
│  │   body)     │    │             │    │                         │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│                         向量化阶段                                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  BGE-M3 向量化 → 5个维度向量 (各1024维)                     │   │
│  │  semantic_vector, domain_vector, method_vector, ...         │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

### 2.2 五维度相似度权重

| 维度 | 权重 | 说明 |
|------|------|------|
| 文本相似度 (semantic) | 30% | 基于语义向量的相似度 |
| 技术领域相似度 (domain) | 25% | 技术分类/领域的匹配程度 |
| 方法相似度 (method) | 20% | 技术方法/实现手段的相似度 |
| 应用场景相似度 (application) | 15% | 应用领域/使用场景的匹配 |
| 创新点相似度 (innovation) | 10% | 核心创新点/关键技术的相似度 |

---

## 三、LLM Prompt 设计

### 3.1 技术领域提取

```python
DOMAIN_EXTRACT_PROMPT = """
请从以下科技成果文本中，提取其所属的技术领域关键词。
要求：
- 提取3-5个最核心的技术领域术语
- 按重要性排序，用逗号分隔
- 只输出关键词，不要解释

文本内容：
{text}

技术领域关键词："""
```

### 3.2 技术方法提取

```python
METHOD_EXTRACT_PROMPT = """
请从以下科技成果文本中，提取其采用的核心技术方法和实现手段。
要求：
- 提取3-5个最关键的技术方法
- 按重要性排序，用逗号分隔
- 只输出方法名称，不要解释

文本内容：
{text}

技术方法："""
```

### 3.3 应用场景提取

```python
APPLICATION_EXTRACT_PROMPT = """
请从以下科技成果文本中，提取其应用场景和使用领域。
要求：
- 提取3-5个主要应用场景
- 按重要性排序，用逗号分隔
- 只输出场景名称，不要解释

文本内容：
{text}

应用场景："""
```

### 3.4 创新点提取

```python
INNOVATION_EXTRACT_PROMPT = """
请从以下科技成果文本中，提取其核心创新点和关键技术突破。
要求：
- 提取2-3个最核心的创新点
- 按重要性排序，用逗号分隔
- 只输出创新点描述，不要解释

文本内容：
{text}

创新点："""
```

---

## 四、多级特征提取器

### 4.1 核心类设计

```python
class MultiLevelFeatureExtractor:
    """多级特征提取器 - LLM提取 + BGE-M3向量化"""

    def __init__(self):
        self.llm_client = AsyncOpenAI(...)  # Qwen3 或其他 LLM
        self.embedding_client = EmbeddingClient()  # BGE-M3

        self.prompts = {
            "domain": DOMAIN_EXTRACT_PROMPT,
            "method": METHOD_EXTRACT_PROMPT,
            "application": APPLICATION_EXTRACT_PROMPT,
            "innovation": INNOVATION_EXTRACT_PROMPT
        }

    async def extract_dimension_text(self, text: str, dim: str) -> str:
        """用LLM提取指定维度的文本内容"""
        prompt = self.prompts[dim].format(text=text[:4000])

        response = await self.llm_client.chat.completions.create(
            model="qwen3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )

        return response.choices[0].message.content.strip()

    async def extract_all_features(self, title: str, body: str) -> Dict:
        """提取所有维度的特征向量"""
        full_text = f"{title} {body}"

        # 1. 语义向量（直接向量化原始文本）
        semantic_vector, _ = await self.embedding_client.get_embedding(full_text)

        # 2. LLM提取各维度文本
        domain_text = await self.extract_dimension_text(full_text, "domain")
        method_text = await self.extract_dimension_text(full_text, "method")
        application_text = await self.extract_dimension_text(full_text, "application")
        innovation_text = await self.extract_dimension_text(full_text, "innovation")

        # 3. 各维度向量化
        domain_vector, _ = await self.embedding_client.get_embedding(domain_text)
        method_vector, _ = await self.embedding_client.get_embedding(method_text)
        application_vector, _ = await self.embedding_client.get_embedding(application_text)
        innovation_vector, _ = await self.embedding_client.get_embedding(innovation_text)

        return {
            "semantic_vector": semantic_vector,
            "domain_vector": domain_vector,
            "method_vector": method_vector,
            "application_vector": application_vector,
            "innovation_vector": innovation_vector,
            "extracted_texts": {
                "domain": domain_text,
                "method": method_text,
                "application": application_text,
                "innovation": innovation_text
            }
        }
```

---

## 五、Milvus 存储结构升级

### 5.1 Collection Schema

```python
fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
    FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=256),

    # 多维度向量
    FieldSchema(name="semantic_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="domain_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="method_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="application_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="innovation_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),

    # 提取的文本内容（用于展示和解释）
    FieldSchema(name="extracted_domain", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="extracted_method", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="extracted_application", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="extracted_innovation", dtype=DataType.VARCHAR, max_length=500),

    # 元数据
    FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="maturity", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="metadata", dtype=DataType.JSON),
]
```

### 5.2 索引配置

```python
# 语义向量索引（用于粗召回）
dense_index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200}
}
collection.create_index(field_name="semantic_vector", index_params=dense_index_params)

# 其他维度向量索引（可选，用于特定场景检索）
for dim in ["domain", "method", "application", "innovation"]:
    collection.create_index(
        field_name=f"{dim}_vector",
        index_params=dense_index_params
    )
```

---

## 六、多级相似度计算

### 6.1 相似度计算器

```python
class MultiLevelSimilarityCalculator:
    """多级相似度计算器"""

    def __init__(self):
        self.weights = {
            "semantic": 0.30,
            "domain": 0.25,
            "method": 0.20,
            "application": 0.15,
            "innovation": 0.10
        }

    def calculate(self, query_vectors: Dict, target: Dict) -> Dict:
        """计算综合相似度"""
        scores = {}
        dimensions = ["semantic", "domain", "method", "application", "innovation"]

        # 1. 各维度独立计算
        for dim in dimensions:
            query_vec = query_vectors[f"{dim}_vector"]
            target_vec = target[f"{dim}_vector"]
            scores[dim] = cosine_similarity(query_vec, target_vec)

        # 2. 加权融合
        total_score = sum(
            scores[dim] * self.weights[dim]
            for dim in self.weights
        )

        # 3. 生成解释
        explanation = self.generate_explanation(scores, target)

        return {
            "total_score": total_score,
            "dimension_scores": scores,
            "explanation": explanation,
            "extracted_info": {
                "domain": target.get("extracted_domain"),
                "method": target.get("extracted_method"),
                "application": target.get("extracted_application"),
                "innovation": target.get("extracted_innovation")
            }
        }

    def generate_explanation(self, scores: Dict, target: Dict) -> str:
        """生成相似度解释"""
        parts = []

        if scores["domain"] > 0.8:
            parts.append(f"技术领域高度匹配：{target.get('extracted_domain', '')}")
        elif scores["domain"] > 0.6:
            parts.append(f"技术领域部分匹配")

        if scores["method"] > 0.7:
            parts.append(f"技术方法相似")

        if scores["application"] > 0.7:
            parts.append(f"应用场景相近")

        return "；".join(parts) if parts else "基于语义相似度匹配"
```

---

## 七、两阶段检索流程

### 7.1 检索架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      第一阶段：粗召回                              │
│                                                                   │
│  查询文本 → BGE-M3 → semantic_vector → Milvus检索 → 候选集(N*3)  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      第二阶段：精排                               │
│                                                                   │
│  1. 对候选集每条数据，取出5个维度向量                              │
│  2. 分别计算5个维度的相似度                                        │
│  3. 加权融合得到 total_score                                       │
│  4. 按 total_score 排序，返回 Top-K                               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 检索实现

```python
async def two_stage_search(
    self,
    query: str,
    resource_type: str,
    top_k: int = 10,
    use_cache: bool = True
) -> List[Dict]:
    """两阶段检索"""

    # ===== 第一阶段：粗召回 =====
    # 1. 提取查询的多维度特征
    query_features = await self.feature_extractor.extract_all_features(query, "")
    query_vectors = {
        "semantic_vector": query_features["semantic_vector"],
        "domain_vector": query_features["domain_vector"],
        "method_vector": query_features["method_vector"],
        "application_vector": query_features["application_vector"],
        "innovation_vector": query_features["innovation_vector"]
    }

    # 2. 用语义向量进行粗召回（扩大3倍）
    candidates = await self.milvus_client.search(
        resource_type=resource_type,
        vector=query_vectors["semantic_vector"],
        top_k=top_k * 3
    )

    # ===== 第二阶段：精排 =====
    # 3. 多级相似度计算
    results = []
    for candidate in candidates:
        similarity = self.similarity_calculator.calculate(
            query_vectors=query_vectors,
            target=candidate
        )
        results.append({
            "id": candidate["id"],
            "source_id": candidate["source_id"],
            "total_score": similarity["total_score"],
            "dimension_scores": similarity["dimension_scores"],
            "explanation": similarity["explanation"],
            "extracted_info": similarity["extracted_info"],
            "metadata": candidate.get("metadata")
        })

    # 4. 按综合分数排序
    results.sort(key=lambda x: x["total_score"], reverse=True)

    return results[:top_k]
```

---

## 八、API 响应格式

### 8.1 匹配接口响应

```json
{
  "request_id": "req_123456",
  "query": "纳米材料涂层",
  "results": [
    {
      "id": "doc_001",
      "source_id": "tec_001",
      "total_score": 0.85,
      "dimension_scores": {
        "semantic": 0.88,
        "domain": 0.90,
        "method": 0.82,
        "application": 0.78,
        "innovation": 0.75
      },
      "explanation": "技术领域高度匹配：纳米材料, 涂层技术；技术方法相似",
      "extracted_info": {
        "domain": "纳米材料, 涂层技术, 表面工程",
        "method": "化学气相沉积, 溶胶凝胶法",
        "application": "航空航天, 汽车制造, 电子设备",
        "innovation": "超疏水性能, 高耐磨性"
      },
      "metadata": {
        "title": "纳米改性透明加硬涂层技术",
        "body": "..."
      }
    }
  ],
  "meta": {
    "total_candidates": 30,
    "retrieval_time_ms": 150
  }
}
```

---

## 九、文件修改清单

| 文件 | 修改内容 |
|------|----------|
| `api/core/feature_extractor.py` | **新建** - 多级特征提取器 |
| `api/core/milvus_client.py` | 升级 Collection Schema，支持5维向量 |
| `api/core/matcher.py` | 实现两阶段检索 + 多级相似度计算 |
| `api/core/similarity_calculator.py` | **新建** - 多级相似度计算器 |
| `api/models/schemas.py` | 新增多级相似度响应模型 |
| `api/routers/data.py` | 数据导入时提取多维度特征 |
| `tools/import_csv.py` | 导入时提取多维度向量 |

---

## 十、实施步骤

### 步骤1：创建多级特征提取器
- 创建 `api/core/feature_extractor.py`
- 实现 LLM 提取 + BGE-M3 向量化逻辑

### 步骤2：创建多级相似度计算器
- 创建 `api/core/similarity_calculator.py`
- 实现五维度相似度计算和加权融合

### 步骤3：升级 Milvus Schema
- 修改 `api/core/milvus_client.py`
- 新增5个维度向量字段和提取文本字段
- 创建新的 Collection（需要重建索引）

### 步骤4：升级检索逻辑
- 修改 `api/core/matcher.py`
- 实现两阶段检索流程

### 步骤5：升级数据导入
- 修改 `tools/import_csv.py`
- 导入时提取多维度特征

### 步骤6：升级 API 响应
- 修改 `api/models/schemas.py`
- 新增多级相似度响应模型

---

**文档创建时间**: 2026-02-10
