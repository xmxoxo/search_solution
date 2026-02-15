import asyncio
from typing import Dict, List, Optional
from api.utils.llm_client import g_llm
from api.core.embedding import g_embedding
from api.utils.logger import g_logger

MULTI_DIMENSION_EXTRACT_PROMPT = """
你是一个专业的文本分析助手，需要从用户提供的文本中提取技术信息。

【分析维度】
1. domain（技术领域）：文本所属的技术领域关键词
2. method（技术方法）：文本采用的核心技术方法和实现手段
3. application（应用场景）：文本的应用场景和使用领域
4. innovation（创新点）：文本的创新点和技术突破

【输出要求】
1. 只基于用户提供的文本内容进行提取，严禁编造或添加文本中不存在的内容
2. 如果某个维度在文本中没有明确提到，输出"无"
3. 只输出关键词，不要任何解释或说明性文字
4. 关键词之间用中文逗号"，"分隔
5. 以纯文本JSON格式输出，禁止添加任何其它解释性文字与内容，输出格式如下：
{{"domain":"关键词1，关键词2","method":"关键词1，关键词2","application":"关键词1，关键词2","innovation":"关键词1，关键词2"}}

【示例】
输入："你好"
输出：{{"domain":"无","method":"无","application":"无","innovation":"无"}}

输入："人工智能在医疗影像诊断中的应用研究，采用深度学习算法进行图像识别"
输出：{{"domain":"人工智能，医疗影像诊断","method":"深度学习算法，图像识别","application":"医疗健康","innovation":"无"}}

输入："本研究首创了多光谱动态调控技术，突破了传统方法的局限性，可应用于农业和医疗领域"
输出：{{"domain":"多光谱技术","method":"多光谱动态调控技术","application":"农业，医疗","innovation":"多光谱动态调控技术"}}

文本内容：
{text}

输出："""

DIMENSIONS = ["domain", "method", "application", "innovation"]
VECTOR_DIM = 1024
EMPTY_VECTOR = [0.0] * VECTOR_DIM

def is_empty_value(value: str) -> bool:
    """判断值是否为空"""
    return not value or value.strip() == "" or value.strip() == "无"

class FeatureExtractor:
    _instance: Optional["FeatureExtractor"] = None

    def __new__(cls) -> "FeatureExtractor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def extract_multi_dimension(self, text: str) -> Dict[str, str]:
        """一次LLM调用提取所有4个维度的文本内容"""
        prompt = MULTI_DIMENSION_EXTRACT_PROMPT.format(text=text[:4000])
        
        try:
            result = await g_llm.parse_json(prompt=prompt, temperature=0.1, max_tokens=300)
            
            extracted = {}
            for dim in DIMENSIONS:
                value = result.get(dim, "")
                if is_empty_value(value):
                    value = ""
                extracted[dim] = value
            
            g_logger.debug(f"Extracted: {extracted}")
            return extracted
        except Exception as e:
            g_logger.warning(f"Failed to extract multi-dimension: {e}")
            return {dim: "" for dim in DIMENSIONS}

    async def extract_all_features(self, title: str, body: str) -> Dict:
        """提取所有维度的特征向量，空值不向量化"""
        full_text = f"{title} {body}"

        semantic_vector, _ = await g_embedding.get_embedding(full_text, use_cache=False)
        
        # 提取特征文本
        extracted_texts = await self.extract_multi_dimension(full_text)

        texts_to_embed = []
        embed_indices = {}
        
        for dim in DIMENSIONS:
            text = extracted_texts.get(dim, "")
            if not is_empty_value(text):
                embed_indices[dim] = len(texts_to_embed)
                texts_to_embed.append(text)
        
        dim_vectors = {dim: EMPTY_VECTOR for dim in DIMENSIONS}
        
        if texts_to_embed:
            embeddings = await g_embedding.get_embeddings_batch(texts_to_embed, use_cache=False)
            for dim, idx in embed_indices.items():
                dim_vectors[dim] = embeddings[idx][0]

        return {
            "semantic_vector": semantic_vector,
            "domain_vector": dim_vectors["domain"],
            "method_vector": dim_vectors["method"],
            "application_vector": dim_vectors["application"],
            "innovation_vector": dim_vectors["innovation"],
            "extracted_texts": extracted_texts
        }

    async def extract_all_features_batch(self, items: List[Dict]) -> List[Dict]:
        """批量提取多维度特征，空值不向量化"""
        if not items:
            return []
        
        full_texts = []
        for item in items:
            full_text = f"{item.get('title', '')} {item.get('body', '')}"
            full_texts.append(full_text)
        
        semantic_embeddings = await g_embedding.get_embeddings_batch(full_texts, use_cache=False)
        
        llm_tasks = [self.extract_multi_dimension(text) for text in full_texts]
        llm_results = await asyncio.gather(*llm_tasks)
        
        texts_to_embed = []
        embed_map = []
        
        for i, extracted in enumerate(llm_results):
            for dim in DIMENSIONS:
                text = extracted.get(dim, "")
                if not is_empty_value(text):
                    embed_map.append((i, dim))
                    texts_to_embed.append(text)
        
        dim_vectors_cache = {}
        for i in range(len(items)):
            dim_vectors_cache[i] = {dim: EMPTY_VECTOR for dim in DIMENSIONS}
        
        if texts_to_embed:
            embeddings = await g_embedding.get_embeddings_batch(texts_to_embed, use_cache=False)
            for idx, (item_idx, dim) in enumerate(embed_map):
                dim_vectors_cache[item_idx][dim] = embeddings[idx][0]
        
        results = []
        for i, item in enumerate(items):
            semantic_vector, _ = semantic_embeddings[i]
            dim_vectors = dim_vectors_cache[i]
            
            results.append({
                "semantic_vector": semantic_vector,
                "domain_vector": dim_vectors["domain"],
                "method_vector": dim_vectors["method"],
                "application_vector": dim_vectors["application"],
                "innovation_vector": dim_vectors["innovation"],
                "extracted_texts": llm_results[i]
            })
        
        return results

g_feature_extractor = FeatureExtractor()

__all__ = ["g_feature_extractor", "FeatureExtractor", "is_empty_value"]
