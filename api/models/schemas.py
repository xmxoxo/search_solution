from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from config.app_config import RESOURCE_TYPES, DEFAULT_TOP_K, DEFAULT_USE_HYBRID

class MatchRequest(BaseModel):
    query: str = Field(..., description="用户查询文本", min_length=1)
    resource_types: Optional[List[str]] = Field(
        default=None,
        description=f"资源类型列表，可选: {RESOURCE_TYPES}"
    )
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=100, description="返回结果数量")
    use_hybrid: bool = Field(default=DEFAULT_USE_HYBRID, description="是否启用混合检索")
    use_cache: bool = Field(default=True, description="是否使用缓存")

class ParsedQueryFilters(BaseModel):
    region: Optional[List[str]] = None
    maturity_gte: Optional[str] = None
    resource_types: Optional[List[str]] = None

class ParsedQueryResult(BaseModel):
    intent_text: str = ""
    filters: ParsedQueryFilters = ParsedQueryFilters()

class MatchResultItem(BaseModel):
    id: str
    source_id: str
    score: float
    region: Optional[str] = None
    maturity: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MatchResponse(BaseModel):
    request_id: Optional[str] = None
    query_parsed: Optional[ParsedQueryResult] = None
    results_by_type: Dict[str, List[MatchResultItem]] = {}
    meta: Optional[Dict[str, Any]] = None

class ParseQueryRequest(BaseModel):
    query: str = Field(..., description="用户查询文本", min_length=1)

class ParseQueryResponse(BaseModel):
    intent_text: str = ""
    filters: Dict[str, Any] = {}

class DataInsertItem(BaseModel):
    id: str = Field(..., description="数据唯一ID")
    fields: Dict[str, Any] = Field(..., description="数据字段")
    raw_text_for_embedding: Optional[str] = Field(
        default=None,
        description="用于向量化的原始文本，如未提供则自动拼接关键字段"
    )

class DataInsertRequest(BaseModel):
    resource_type: str = Field(
        ...,
        description=f"资源类型，可选: {RESOURCE_TYPES}"
    )
    items: List[DataInsertItem] = Field(..., description="待插入的数据列表")
    async_process: bool = Field(default=False, description="是否异步处理")

class DataInsertResponse(BaseModel):
    ingested_count: int = 0
    failed_ids: List[str] = []
    task_id: Optional[str] = None

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None

class CollectionStats(BaseModel):
    exists: bool
    collection_name: Optional[str] = None
    count: int = 0
    indexes: List[Dict[str, Any]] = []

class StatsResponse(BaseModel):
    resource_types: List[str]
    collections: Dict[str, CollectionStats]
    total_count: int
