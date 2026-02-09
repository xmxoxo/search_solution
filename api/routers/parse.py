from fastapi import APIRouter, Request, HTTPException
from api.models.schemas import (
    ParseQueryRequest,
    ParseQueryResponse,
    ErrorResponse
)
from api.core.query_parser import g_query_parser
from api.utils.logger import g_logger

router = APIRouter()

@router.post(
    "/parse-query",
    response_model=ParseQueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "无效请求"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"}
    }
)
async def parse_query(
    request: Request,
    body: ParseQueryRequest
):
    """
    查询解析接口（内部调试用）
    
    用于调试LLM结构化抽取效果，
    展示从用户查询中提取的意图关键词和结构化条件。
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    try:
        parsed = await g_query_parser.parse(body.query, use_cache=False)
        
        g_logger.info(f"[{request_id}] Query parsed successfully")
        
        return ParseQueryResponse(
            intent_text=parsed.intent_text,
            filters=parsed.filters.model_dump() if hasattr(parsed.filters, "model_dump") else dict(parsed.filters)
        )
        
    except Exception as e:
        g_logger.error(f"[{request_id}] Parse error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": str(e)}
        )
