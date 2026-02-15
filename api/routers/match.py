import time
from typing import Annotated
from fastapi import APIRouter, Request, HTTPException
from api.models.schemas import (
    MatchRequest,
    MatchResponse,
    MatchResultItem,
    ParsedQueryResult,
    QueryFeatures,
    ErrorResponse
)
from api.core.matcher import g_matcher
from api.utils.logger import g_logger

router = APIRouter()

@router.post(
    "/match",
    response_model=MatchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "无效请求"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"}
    }
)
async def match_resources(
    request: Request,
    body: MatchRequest
):
    """
    数据匹配接口
    
    根据用户查询文本，在指定的资源类型中进行语义匹配检索。
    支持混合检索（dense + sparse），兼顾召回率与准确率。
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    try:
        start_time = time.time()
        
        result = await g_matcher.match(
            query=body.query,
            resource_types=body.resource_types,
            top_k=body.top_k,
            use_cache=body.use_cache,
            use_multi_level=body.use_multi_level
        )
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        results_by_type = {}
        total_candidates = 0
        
        for resource_type, items in result.get("results_by_type", {}).items():
            results_by_type[resource_type] = [
                MatchResultItem(**item) for item in items
            ]
            total_candidates += len(items)
        
        query_parsed = None
        if "query_parsed" in result:
            query_parsed = ParsedQueryResult(**result["query_parsed"])
        
        query_features = None
        if "query_features" in result:
            query_features = QueryFeatures(**result["query_features"])
        
        response = MatchResponse(
            request_id=request_id,
            query_parsed=query_parsed,
            results_by_type=results_by_type,
            query_features=query_features,
            meta={
                "total_candidates": total_candidates,
                "retrieval_time_ms": elapsed_ms,
                "use_hybrid": True,
                "use_multi_level": body.use_multi_level
            }
        )
        
        g_logger.info(
            f"[{request_id}] Match completed: {total_candidates} results in {elapsed_ms}ms"
        )
        
        return response
        
    except ValueError as e:
        g_logger.error(f"[{request_id}] Invalid request: {e}")
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_request", "message": str(e)}
        )
    except Exception as e:
        g_logger.error(f"[{request_id}] Match error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": str(e)}
        )
