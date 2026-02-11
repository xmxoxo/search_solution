from typing import Any, Dict, List, Optional
from api.core.embedding import g_embedding
from api.core.query_parser import g_query_parser
from api.core.milvus_client import g_milvus
from api.utils.logger import g_logger
from api.utils.cache import g_cache, CacheKeys, CacheTTL
from config.app_config import RESOURCE_TYPES, DEFAULT_TOP_K, DEFAULT_USE_HYBRID, MIN_SCORE_THRESHOLD

class MatchResult:
    def __init__(
        self,
        id: str,
        source_id: str,
        score: float,
        region: Optional[str] = None,
        maturity: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = id
        self.source_id = source_id
        self.score = score
        self.region = region
        self.maturity = maturity
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "score": self.score,
            "region": self.region,
            "maturity": self.maturity,
            "metadata": self.metadata
        }

class Matcher:
    _instance: Optional["Matcher"] = None
    
    def __new__(cls) -> "Matcher":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def match(
        self,
        query: str,
        resource_types: Optional[List[str]] = None,
        top_k: int = DEFAULT_TOP_K,
        use_hybrid: bool = DEFAULT_USE_HYBRID,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        if resource_types is None:
            resource_types = RESOURCE_TYPES
        
        resource_types = [rt for rt in resource_types if rt in RESOURCE_TYPES]
        if not resource_types:
            resource_types = RESOURCE_TYPES
        
        cache_key = g_cache.generate_key(
            CacheKeys.MATCH_RESULT,
            f"{query}:{':'.join(resource_types)}:{top_k}:{use_hybrid}"
        )
        
        if use_cache:
            cached = await g_cache.get_json(cache_key)
            if cached:
                return cached
        
        try:
            parsed = await g_query_parser.parse(query, use_cache=use_cache)
            
            dense_vector, sparse_vector = await g_embedding.get_embedding(
                parsed.intent_text,
                use_cache=use_cache
            )
            
            filter_expr = g_milvus.build_filter_expr(parsed.filters)
            
            results_by_type = {}
            for resource_type in resource_types:
                try:
                    results = await g_milvus.hybrid_search(
                        resource_type=resource_type,
                        dense_vector=dense_vector,
                        sparse_vector=sparse_vector,
                        filter_expr=filter_expr,
                        top_k=top_k,
                        use_hybrid=use_hybrid
                    )
                    filtered_results = [
                        r for r in results if r.get("score", 0) >= MIN_SCORE_THRESHOLD
                    ]
                    results_by_type[resource_type] = filtered_results
                    g_logger.info(
                        f"{resource_type}: {len(results)} results, {len(filtered_results)} after threshold filter"
                    )
                except Exception as e:
                    g_logger.warning(f"Search failed for {resource_type}: {e}")
                    results_by_type[resource_type] = []
            
            response = {
                "query_parsed": parsed.model_dump(),
                "results_by_type": results_by_type
            }
            
            if use_cache:
                await g_cache.setex_json(cache_key, CacheTTL.MATCH_RESULT, response)
            
            return response
        except Exception as e:
            g_logger.error(f"Match error: {e}")
            raise

g_matcher = Matcher()

__all__ = ["g_matcher", "Matcher", "MatchResult"]
