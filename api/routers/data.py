import uuid
from typing import Annotated
from fastapi import APIRouter, Request, HTTPException
from api.models.schemas import (
    DataInsertRequest,
    DataInsertResponse,
    ErrorResponse
)
from api.core.embedding import g_embedding
from api.core.milvus_client import g_milvus
from api.utils.logger import g_logger
from config.app_config import RESOURCE_TYPES

router = APIRouter()

def generate_embedding_text(item: dict, resource_type: str) -> str:
    """根据资源类型生成用于向量化的文本"""
    fields = item.get("fields", {})
    parts = []
    
    if resource_type == "expert":
        if "name" in fields:
            parts.append(fields["name"])
        if "research_direction" in fields:
            parts.append(fields["research_direction"])
        if "introduction" in fields:
            parts.append(fields["introduction"])
        if "specialty" in fields:
            parts.append(fields["specialty"])
    
    elif resource_type == "project":
        if "name" in fields:
            parts.append(fields["name"])
        if "description" in fields:
            parts.append(fields["description"])
        if "summary" in fields:
            parts.append(fields["summary"])
    
    elif resource_type == "patent":
        if "title" in fields:
            parts.append(fields["title"])
        if "abstract" in fields:
            parts.append(fields["abstract"])
        if "summary" in fields:
            parts.append(fields["summary"])
    
    elif resource_type == "enterprise":
        if "name" in fields:
            parts.append(fields["name"])
        if "business_scope" in fields:
            parts.append(fields["business_scope"])
        if "products" in fields:
            parts.append(fields["products"])
    
    elif resource_type == "paper":
        if "title" in fields:
            parts.append(fields["title"])
        if "abstract" in fields:
            parts.append(fields["abstract"])
    
    elif resource_type == "institution":
        if "name" in fields:
            parts.append(fields["name"])
        if "research_focus" in fields:
            parts.append(fields["research_focus"])
    
    elif resource_type == "tec":
        if "name" in fields:
            parts.append(fields["name"])
        if "description" in fields:
            parts.append(fields["description"])
        if "summary" in fields:
            parts.append(fields["summary"])
    
    return " ".join(parts)

@router.post(
    "/data/insert",
    response_model=DataInsertResponse,
    responses={
        400: {"model": ErrorResponse, "description": "无效请求"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"}
    }
)
async def insert_data(
    request: Request,
    body: DataInsertRequest
):
    """
    数据导入接口
    
    用于成果、专家、专利等资源的单条或批量导入。
    自动生成向量并写入Milvus。
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    if body.resource_type not in RESOURCE_TYPES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_resource_type",
                "message": f"资源类型必须是以下之一: {RESOURCE_TYPES}"
            }
        )
    
    ingested_count = 0
    failed_ids = []
    task_id = None
    
    if body.async_process:
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        g_logger.info(f"[{request_id}] Async processing started, task_id: {task_id}")
    
    try:
        for item in body.items:
            try:
                raw_text = item.raw_text_for_embedding
                if not raw_text:
                    raw_text = generate_embedding_text(
                        {"fields": item.fields},
                        body.resource_type
                    )
                
                if not raw_text or not raw_text.strip():
                    g_logger.warning(f"[{request_id}] Empty embedding text for item {item.id}")
                    failed_ids.append(item.id)
                    continue
                
                dense_vector, sparse_vector = await g_embedding.get_embedding(
                    raw_text,
                    use_cache=False
                )
                
                region = item.fields.get("region", "")
                maturity = item.fields.get("maturity", "")
                
                await g_milvus.insert(
                    resource_type=body.resource_type,
                    id=item.id,
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    source_id=item.fields.get("source_id", item.id),
                    region=region,
                    maturity=maturity,
                    metadata=item.fields
                )
                
                ingested_count += 1
                
            except Exception as e:
                g_logger.error(f"[{request_id}] Failed to insert item {item.id}: {e}")
                failed_ids.append(item.id)
        
        g_logger.info(
            f"[{request_id}] Data insert completed: {ingested_count} success, {len(failed_ids)} failed"
        )
        
        return DataInsertResponse(
            ingested_count=ingested_count,
            failed_ids=failed_ids,
            task_id=task_id
        )
        
    except Exception as e:
        g_logger.error(f"[{request_id}] Data insert error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": str(e)}
        )
