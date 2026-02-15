import uuid
from typing import Annotated
from fastapi import APIRouter, Request, HTTPException
from api.models.schemas import (
    DataInsertRequest,
    DataInsertResponse,
    ErrorResponse,
    StatsResponse,
    CollectionStats
)
from api.core.embedding import g_embedding
from api.core.milvus_client import g_milvus
from api.core.feature_extractor import g_feature_extractor
from api.utils.logger import g_logger
from config.app_config import RESOURCE_TYPES

router = APIRouter()

def generate_embedding_text(item: dict, resource_type: str) -> str:
    """根据资源类型生成用于向量化的文本"""
    fields = item.get("fields", {})
    parts = []
    
    if "title" in fields:
        parts.append(fields["title"])
    
    if resource_type == "expert":
        if "name" in fields and "title" not in fields:
            parts.append(fields["name"])
        if "research_direction" in fields:
            parts.append(fields["research_direction"])
        if "introduction" in fields:
            parts.append(fields["introduction"])
        if "specialty" in fields:
            parts.append(fields["specialty"])
    
    elif resource_type == "project":
        if "name" in fields and "title" not in fields:
            parts.append(fields["name"])
        if "description" in fields:
            parts.append(fields["description"])
        if "summary" in fields:
            parts.append(fields["summary"])
    
    elif resource_type == "patent":
        if "title" in fields:
            pass
        elif "name" in fields:
            parts.append(fields["name"])
        if "abstract" in fields:
            parts.append(fields["abstract"])
        if "summary" in fields:
            parts.append(fields["summary"])
    
    elif resource_type == "enterprise":
        if "name" in fields and "title" not in fields:
            parts.append(fields["name"])
        if "business_scope" in fields:
            parts.append(fields["business_scope"])
        if "products" in fields:
            parts.append(fields["products"])
    
    elif resource_type == "paper":
        if "title" in fields:
            pass
        elif "name" in fields:
            parts.append(fields["name"])
        if "abstract" in fields:
            parts.append(fields["abstract"])
    
    elif resource_type == "institution":
        if "name" in fields and "title" not in fields:
            parts.append(fields["name"])
        if "research_focus" in fields:
            parts.append(fields["research_focus"])
    
    elif resource_type == "tec":
        if "name" in fields and "title" not in fields:
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
        v1_insert_items = []
        v2_insert_items = []
        
        for item in body.items:
            try:
                title = item.fields.get("title", "")
                body_text = item.fields.get("body", "")
                
                if not body_text:
                    g_logger.warning(f"[{request_id}] Empty body text for item {item.id}")
                    failed_ids.append(item.id)
                    continue
                
                # 生成v1混合检索向量
                full_text = f"{title} {body_text}"
                dense_vector, sparse_vector = await g_embedding.get_embedding(full_text, use_cache=False)
                
                # 处理 sparse_vector 格式
                if sparse_vector is None:
                    sparse_vector = {}
                elif hasattr(sparse_vector, 'indices') and hasattr(sparse_vector, 'values'):
                    sparse_vector = dict(zip(sparse_vector.indices, sparse_vector.values))
                elif not isinstance(sparse_vector, dict):
                    sparse_vector = {}
                
                # 生成v2扩展维度向量
                features = await g_feature_extractor.extract_all_features(title, body_text)
                
                region = item.fields.get("region", "")
                maturity = item.fields.get("maturity", "")
                
                # v1数据（混合检索）
                v1_insert_items.append({
                    'id': item.id,
                    'dense_vector': dense_vector,
                    'sparse_vector': sparse_vector,
                    'source_id': item.fields.get("source_id", item.id),
                    'region': region,
                    'maturity': maturity,
                    'metadata': item.fields
                })
                
                # v2数据（扩展维度）
                v2_insert_items.append({
                    'id': item.id,
                    'vectors': {
                        'domain_vector': features['domain_vector'],
                        'method_vector': features['method_vector'],
                        'application_vector': features['application_vector'],
                        'innovation_vector': features['innovation_vector']
                    },
                    'extracted_texts': features['extracted_texts'],
                    'source_id': item.fields.get("source_id", item.id),
                    'region': region,
                    'maturity': maturity,
                    'metadata': item.fields
                })
                
                ingested_count += 1
                
            except Exception as e:
                g_logger.error(f"[{request_id}] Failed to process item {item.id}: {e}")
                failed_ids.append(item.id)
        
        # 插入v1数据（混合检索）
        if v1_insert_items:
            inserted_count = await g_milvus.insert_batch(
                resource_type=body.resource_type,
                items=v1_insert_items,
                skip_duplicate_check=True
            )
            g_logger.info(f"[{request_id}] Inserted {inserted_count} items into v1 collection")
        
        # 插入v2数据（扩展维度）
        if v2_insert_items:
            inserted_count = await g_milvus.insert_batch_v2(
                resource_type=body.resource_type,
                items=v2_insert_items,
                skip_duplicate_check=True
            )
            g_logger.info(f"[{request_id}] Inserted {inserted_count} items into v2 collections")
        
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

@router.get(
    "/data/stats",
    response_model=StatsResponse,
    responses={
        500: {"model": ErrorResponse, "description": "服务器内部错误"}
    }
)
async def get_stats(request: Request):
    """
    查询数据统计接口
    
    返回所有资源类型的数据情况，包括数据量和索引状态。
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    try:
        collections = {}
        total_count = 0
        
        for resource_type in RESOURCE_TYPES:
            stats = g_milvus.get_collection_stats(resource_type)
            collections[resource_type] = CollectionStats(**stats)
            total_count += stats.get("count", 0)
        
        g_logger.info(f"[{request_id}] Stats query completed: total {total_count} records")
        
        return StatsResponse(
            resource_types=RESOURCE_TYPES,
            collections=collections,
            total_count=total_count
        )
        
    except Exception as e:
        g_logger.error(f"[{request_id}] Stats query error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": str(e)}
        )
