from typing import Any, Dict, List, Optional
from api.core.embedding import g_embedding
from api.core.query_parser import g_query_parser
from api.core.milvus_client import g_milvus, USE_MULTI_LEVEL
from api.core.feature_extractor import g_feature_extractor
from api.core.similarity_calculator import g_similarity_calculator
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
        metadata: Optional[Dict[str, Any]] = None,
        dimension_scores: Optional[Dict[str, float]] = None,
        explanation: Optional[str] = None,
        extracted_info: Optional[Dict[str, str]] = None
    ):
        self.id = id
        self.source_id = source_id
        self.score = score
        self.region = region
        self.maturity = maturity
        self.metadata = metadata or {}
        self.dimension_scores = dimension_scores or {}
        self.explanation = explanation or ""
        self.extracted_info = extracted_info or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "score": self.score,
            "region": self.region,
            "maturity": self.maturity,
            "metadata": self.metadata,
            "dimension_scores": self.dimension_scores,
            "explanation": self.explanation,
            "extracted_info": self.extracted_info
        }

class Matcher:
    _instance: Optional["Matcher"] = None
    
    def __new__(cls) -> "Matcher":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def match_v2(
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
            f"v2:{query}:{':'.join(resource_types)}:{top_k}:{use_hybrid}"
        )
        
        if use_cache:
            cached = await g_cache.get_json(cache_key)
            if cached:
                return cached
        
        try:
            parsed = await g_query_parser.parse(query, use_cache=use_cache)
            
            # 计算向量用于混合检索
            dense_vector, sparse_vector = await g_embedding.get_embedding(
                query,
                use_cache=use_cache
            )
            
            # 扩展特征提取，包括特征文本
            query_features = await g_feature_extractor.extract_all_features(query, "")
            
            extracted_texts = query_features.get("extracted_texts", {})
            all_empty = all(
                extracted_texts.get(dim, "无") in ("无", "", None)
                for dim in ["domain", "method", "application", "innovation"]
            )
            
            if all_empty:
                g_logger.info(f"Query '{query[:30]}...' has no extracted features, falling back to v1 hybrid search")
                return await self._match_v1_internal(
                    parsed=parsed,
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    resource_types=resource_types,
                    top_k=top_k,
                    use_hybrid=use_hybrid,
                    use_cache=use_cache,
                    cache_key=cache_key
                )
            
            query_vectors = {
                "domain_vector": query_features["domain_vector"],
                "method_vector": query_features["method_vector"],
                "application_vector": query_features["application_vector"],
                "innovation_vector": query_features["innovation_vector"]
            }
            
            filter_expr = g_milvus.build_filter_expr(parsed.filters)
            
            results_by_type = {}
            for resource_type in resource_types:
                try:
                    candidates = await g_milvus.search_v2(
                        resource_type=resource_type,
                        dense_vector=dense_vector,
                        sparse_vector=sparse_vector,
                        filter_expr=filter_expr,
                        top_k=top_k,
                        use_hybrid=use_hybrid
                    )
                    
                    v1_results = await g_milvus.hybrid_search(
                        resource_type=resource_type,
                        dense_vector=dense_vector,
                        sparse_vector=sparse_vector,
                        filter_expr=filter_expr,
                        top_k=top_k,
                        use_hybrid=use_hybrid
                    )
                    
                    v1_result_ids = {r["id"] for r in v1_results}
                    g_logger.info(f"v1_result_ids:{v1_result_ids}")
                    
                    scored_results = []
                    for candidate in candidates:
                        similarity = g_similarity_calculator.calculate(
                            query_vectors=query_vectors,
                            target=candidate,
                            semantic_score=candidate.get("semantic_score", 0.0),
                            query_extracted_texts=query_features.get("extracted_texts")
                        )
                        
                        if similarity["total_score"] >= MIN_SCORE_THRESHOLD:
                            result = MatchResult(
                                id=candidate["id"],
                                source_id=candidate["source_id"],
                                score=similarity["total_score"],
                                region=candidate.get("region"),
                                maturity=candidate.get("maturity"),
                                metadata=candidate.get("metadata"),
                                dimension_scores=similarity["dimension_scores"],
                                explanation=similarity["explanation"],
                                extracted_info=similarity["extracted_info"]
                            )
                            scored_results.append(result)
                    
                    for v1_item in v1_results:
                        if v1_item["id"] not in v1_result_ids:
                            v1_result_ids.add(v1_item["id"])
                            if v1_item.get("score", 0) >= MIN_SCORE_THRESHOLD:
                                result = MatchResult(
                                    id=v1_item["id"],
                                    source_id=v1_item["source_id"],
                                    score=v1_item["score"],
                                    region=v1_item.get("region"),
                                    maturity=v1_item.get("maturity"),
                                    metadata=v1_item.get("metadata"),
                                    dimension_scores={
                                        "semantic": v1_item.get("score", 0),
                                        "domain": 0,
                                        "method": 0,
                                        "application": 0,
                                        "innovation": 0
                                    },
                                    explanation="语义匹配（保底）",
                                    extracted_info={
                                        "domain": "",
                                        "method": "",
                                        "application": "",
                                        "innovation": ""
                                    }
                                )
                                scored_results.append(result)
                    
                    scored_results.sort(key=lambda x: x.score, reverse=True)
                    results_by_type[resource_type] = [r.to_dict() for r in scored_results[:top_k]]
                    g_logger.info(
                        f"{resource_type}: {len(candidates)} v2 candidates, {len(v1_results)} v1 backup, {len(results_by_type[resource_type])} final results"
                    )
                except Exception as e:
                    g_logger.warning(f"Search failed for {resource_type}: {e}")
                    results_by_type[resource_type] = []
            
            response = {
                "query_parsed": parsed.model_dump(),
                "results_by_type": results_by_type,
                "query_features": {
                    "extracted_texts": query_features["extracted_texts"]
                }
            }
            
            if use_cache:
                await g_cache.setex_json(cache_key, CacheTTL.MATCH_RESULT, response)
            
            return response
        except Exception as e:
            g_logger.error(f"Match v2 error: {e}")
            raise
    
    async def _match_v1_internal(
        self,
        parsed: Any,
        dense_vector: List[float],
        sparse_vector: Optional[Dict[int, float]],
        resource_types: List[str],
        top_k: int,
        use_hybrid: bool,
        use_cache: bool,
        cache_key: str
    ) -> Dict[str, Any]:
        """内部方法：执行v1混合检索（用于降级场景）"""
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
            "results_by_type": results_by_type,
            "query_features": {
                "extracted_texts": {"domain": "无", "method": "无", "application": "无", "innovation": "无"},
                "fallback_to_v1": True
            }
        }
        
        if use_cache:
            await g_cache.setex_json(cache_key, CacheTTL.MATCH_RESULT, response)
        
        return response
    
    async def match(
        self,
        query: str,
        resource_types: Optional[List[str]] = None,
        top_k: int = DEFAULT_TOP_K,
        use_hybrid: bool = DEFAULT_USE_HYBRID,
        use_cache: bool = True,
        use_multi_level: bool = USE_MULTI_LEVEL
    ) -> Dict[str, Any]:
        if use_multi_level:
            return await self.match_v2(query, resource_types, top_k, use_hybrid, use_cache)
        
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
                query,
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
