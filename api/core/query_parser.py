from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from api.utils.llm_client import g_llm
from api.utils.logger import g_logger
from api.utils.cache import g_cache, CacheKeys, CacheTTL
from config.app_config import RESOURCE_TYPES, MATURITY_LEVELS

class ParsedQuery(BaseModel):
    intent_text: str = Field(default="", description="核心技术关键词/意图文本")
    filters: Dict[str, Any] = Field(default_factory=dict, description="结构化过滤条件")
    
    class Config:
        extra = "allow"

QUERY_PARSER_SYSTEM_PROMPT = """你是一个科技领域信息抽取专家。
请从用户输入的查询文本中提取核心技术关键词和结构化过滤条件。

注意事项：
1. intent_text 应该是用于语义匹配的核心技术关键词或描述
2. region 必须标准化到市级，如"上海市"、"北京市"、"杭州市"等
3. maturity_gte 只能是以下值之一：研发、小试、中试、量产
4. resource_types 只能从以下列表中选择：expert, project, patent, enterprise, paper, institution, tec
5. 如果用户未提及某个条件，对应字段设为null或不包含"""

QUERY_PARSER_USER_TEMPLATE = """请从以下用户输入中提取信息，输出严格为JSON格式：

输出格式：
{{
  "intent_text": "核心技术关键词（字符串）",
  "filters": {{
    "region": ["上海市", "杭州市", ...] 或 null,
    "maturity_gte": "研发|小试|中试|量产" 或 null,
    "resource_types": ["expert", "project", ...] 或 null
  }}
}}

用户输入：{query}

请输出JSON："""

class QueryParser:
    _instance: Optional["QueryParser"] = None
    
    def __new__(cls) -> "QueryParser":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def parse(
        self,
        query: str,
        use_cache: bool = True
    ) -> ParsedQuery:
        if use_cache:
            cache_key = g_cache.generate_key(CacheKeys.LLM_PARSE, query)
            cached = await g_cache.get_json(cache_key)
            if cached:
                return ParsedQuery(**cached)
        
        try:
            prompt = QUERY_PARSER_USER_TEMPLATE.format(query=query)
            result = await g_llm.parse_json(
                prompt=prompt,
                system_prompt=QUERY_PARSER_SYSTEM_PROMPT,
                temperature=0.0
            )
            
            parsed = ParsedQuery(**result)
            
            if use_cache:
                cache_key = g_cache.generate_key(CacheKeys.LLM_PARSE, query)
                await g_cache.setex_json(
                    cache_key,
                    CacheTTL.LLM_PARSE,
                    parsed.model_dump()
                )
            
            return parsed
        except Exception as e:
            g_logger.error(f"Query parsing error: {e}")
            return ParsedQuery(intent_text=query, filters={})

g_query_parser = QueryParser()

__all__ = ["g_query_parser", "QueryParser", "ParsedQuery"]
