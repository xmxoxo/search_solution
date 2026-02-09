from typing import Any, Dict, List, Optional
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from config.app_config import (
    MILVUS_HOST,
    MILVUS_PORT,
    HYBRID_SEARCH_ALPHA,
    HYBRID_SEARCH_BETA,
    DEFAULT_TOP_K
)
from api.utils.logger import g_logger

DENSE_VECTOR_DIM = 1024

class MilvusClient:
    _instance: Optional["MilvusClient"] = None
    _collections: Dict[str, Collection] = {}
    
    def __new__(cls) -> "MilvusClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connected = False
        return cls._instance
    
    def connect(self, alias: str = "default"):
        if not self._connected:
            try:
                connections.connect(
                    alias=alias,
                    host=MILVUS_HOST,
                    port=MILVUS_PORT
                )
                self._connected = True
                g_logger.info(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
            except Exception as e:
                g_logger.error(f"Failed to connect to Milvus: {e}")
                raise
    
    def disconnect(self, alias: str = "default"):
        if self._connected:
            connections.disconnect(alias)
            self._connected = False
            self._collections.clear()
    
    def _get_collection_schema(self, resource_type: str) -> CollectionSchema:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=DENSE_VECTOR_DIM),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="maturity", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        return CollectionSchema(fields=fields, description=f"Collection for {resource_type}")
    
    def _create_indexes(self, collection: Collection):
        dense_index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index(
            field_name="dense_vector",
            index_params=dense_index_params
        )
        
        sparse_index_params = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "params": {"drop_ratio_build": 0.2}
        }
        collection.create_index(
            field_name="sparse_vector",
            index_params=sparse_index_params
        )
    
    def get_or_create_collection(self, resource_type: str) -> Collection:
        if resource_type in self._collections:
            return self._collections[resource_type]
        
        self.connect()
        
        collection_name = f"ime_{resource_type}"
        
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load()
        else:
            schema = self._get_collection_schema(resource_type)
            collection = Collection(name=collection_name, schema=schema)
            self._create_indexes(collection)
            collection.load()
            g_logger.info(f"Created collection: {collection_name}")
        
        self._collections[resource_type] = collection
        return collection
    
    async def insert(
        self,
        resource_type: str,
        id: str,
        dense_vector: List[float],
        sparse_vector: Optional[Dict[int, float]] = None,
        source_id: Optional[str] = None,
        region: Optional[str] = None,
        maturity: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        collection = self.get_or_create_collection(resource_type)
        
        data = {
            "id": id,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector or {},
            "source_id": source_id or "",
            "region": region or "",
            "maturity": maturity or "",
            "metadata": metadata or {}
        }
        
        collection.insert([data])
        collection.flush()
        g_logger.info(f"Inserted document {id} into {resource_type}")
    
    async def hybrid_search(
        self,
        resource_type: str,
        dense_vector: List[float],
        sparse_vector: Optional[Dict[int, float]] = None,
        filter_expr: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        use_hybrid: bool = True,
        alpha: float = HYBRID_SEARCH_ALPHA,
        beta: float = HYBRID_SEARCH_BETA
    ) -> List[Dict[str, Any]]:
        collection = self.get_or_create_collection(resource_type)
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 100}
        }
        
        output_fields = ["id", "source_id", "region", "maturity", "metadata"]
        
        if use_hybrid and sparse_vector:
            dense_results = collection.search(
                data=[dense_vector],
                anns_field="dense_vector",
                param=search_params,
                limit=top_k * 2,
                expr=filter_expr,
                output_fields=output_fields
            )
            
            sparse_search_params = {
                "metric_type": "IP",
                "params": {}
            }
            sparse_results = collection.search(
                data=[sparse_vector],
                anns_field="sparse_vector",
                param=sparse_search_params,
                limit=top_k * 2,
                expr=filter_expr,
                output_fields=output_fields
            )
            
            combined_scores = {}
            for hits in dense_results:
                for hit in hits:
                    doc_id = hit.entity.get("id")
                    combined_scores[doc_id] = {"score": alpha * hit.score, "entity": hit.entity}
            
            for hits in sparse_results:
                for hit in hits:
                    doc_id = hit.entity.get("id")
                    if doc_id in combined_scores:
                        combined_scores[doc_id]["score"] += beta * hit.score
                    else:
                        combined_scores[doc_id] = {"score": beta * hit.score, "entity": hit.entity}
            
            sorted_results = sorted(combined_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
            
            final_results = []
            for item in sorted_results:
                entity = item["entity"]
                final_results.append({
                    "id": entity.get("id"),
                    "source_id": entity.get("source_id"),
                    "score": item["score"],
                    "region": entity.get("region"),
                    "maturity": entity.get("maturity"),
                    "metadata": entity.get("metadata")
                })
            return final_results
        else:
            results = collection.search(
                data=[dense_vector],
                anns_field="dense_vector",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=output_fields
            )
            
            final_results = []
            for hits in results:
                for hit in hits:
                    entity = hit.entity
                    final_results.append({
                        "id": entity.get("id"),
                        "source_id": entity.get("source_id"),
                        "score": hit.score,
                        "region": entity.get("region"),
                        "maturity": entity.get("maturity"),
                        "metadata": entity.get("metadata")
                    })
            return final_results
    
    def build_filter_expr(self, filters: Dict[str, Any]) -> Optional[str]:
        conditions = []
        
        if "region" in filters and filters["region"]:
            regions = [f"'{r}'" for r in filters["region"]]
            conditions.append(f"region in [{', '.join(regions)}]")
        
        if "maturity_gte" in filters and filters["maturity_gte"]:
            maturity_order = ["研发", "小试", "中试", "量产"]
            try:
                idx = maturity_order.index(filters["maturity_gte"])
                allowed = [f"'{m}'" for m in maturity_order[idx:]]
                conditions.append(f"maturity in [{', '.join(allowed)}]")
            except ValueError:
                pass
        
        return " and ".join(conditions) if conditions else None

g_milvus = MilvusClient()

__all__ = ["g_milvus", "MilvusClient"]
