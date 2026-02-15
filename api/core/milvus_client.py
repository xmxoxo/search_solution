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

VECTOR_DIM = 1024

USE_MULTI_LEVEL = True

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
    
    def _get_collection_schema_v2a(self, resource_type: str) -> CollectionSchema:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="domain_vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="method_vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="extracted_domain", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="extracted_method", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="maturity", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        return CollectionSchema(fields=fields, description=f"Multi-level Collection A for {resource_type}")
    
    def _get_collection_schema_v2b(self, resource_type: str) -> CollectionSchema:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="application_vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="innovation_vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="extracted_application", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="extracted_innovation", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="maturity", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        return CollectionSchema(fields=fields, description=f"Multi-level Collection B for {resource_type}")
    
    def _get_collection_schema(self, resource_type: str) -> CollectionSchema:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="source_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="maturity", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        return CollectionSchema(fields=fields, description=f"Collection for {resource_type}")
    
    def _create_indexes_v2a(self, collection: Collection):
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200}
        }
        for field in ["domain_vector", "method_vector"]:
            collection.create_index(field_name=field, index_params=index_params)
    
    def _create_indexes_v2b(self, collection: Collection):
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200}
        }
        for field in ["application_vector", "innovation_vector"]:
            collection.create_index(field_name=field, index_params=index_params)
    
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
    
    def get_or_create_collection(self, resource_type: str, use_multi_level: bool = False, collection_suffix: str = "") -> Collection:
        if collection_suffix:
            cache_key = f"{resource_type}_v2_{collection_suffix}"
            collection_name = f"ime_{resource_type}_v2_{collection_suffix}"
        elif use_multi_level:
            cache_key = f"{resource_type}_v2a"
            collection_name = f"ime_{resource_type}_v2a"
        else:
            cache_key = resource_type
            collection_name = f"ime_{resource_type}"
        
        if cache_key in self._collections:
            return self._collections[cache_key]
        
        self.connect()
        
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load()
        else:
            if collection_suffix == "a":
                schema = self._get_collection_schema_v2a(resource_type)
            elif collection_suffix == "b":
                schema = self._get_collection_schema_v2b(resource_type)
            elif use_multi_level:
                schema = self._get_collection_schema_v2a(resource_type)
            else:
                schema = self._get_collection_schema(resource_type)
            
            collection = Collection(name=collection_name, schema=schema)
            
            if collection_suffix == "a" or use_multi_level:
                self._create_indexes_v2a(collection)
            elif collection_suffix == "b":
                self._create_indexes_v2b(collection)
            else:
                self._create_indexes(collection)
            collection.load()
            g_logger.info(f"Created collection: {collection_name}")
        
        self._collections[cache_key] = collection
        return collection
    
    def get_or_create_collections_v2(self, resource_type: str) -> tuple:
        coll_a = self.get_or_create_collection(resource_type, collection_suffix="a")
        coll_b = self.get_or_create_collection(resource_type, collection_suffix="b")
        return coll_a, coll_b
    
    async def insert_v2(
        self,
        resource_type: str,
        id: str,
        vectors: Dict[str, List[float]],
        extracted_texts: Dict[str, str],
        source_id: Optional[str] = None,
        region: Optional[str] = None,
        maturity: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        skip_duplicate_check: bool = False
    ):
        coll_a, coll_b = self.get_or_create_collections_v2(resource_type)
        
        if not skip_duplicate_check:
            existing = coll_a.query(
                expr=f"id == '{id}'",
                output_fields=["id"],
                limit=1
            )
            if existing:
                g_logger.warning(f"Document {id} already exists, skipping")
                return False
        
        data_a = {
            "id": id,
            "source_id": source_id or id,
            "domain_vector": vectors.get("domain_vector", []),
            "method_vector": vectors.get("method_vector", []),
            "extracted_domain": extracted_texts.get("domain", "")[:500],
            "extracted_method": extracted_texts.get("method", "")[:500],
            "region": region or "",
            "maturity": maturity or "",
            "metadata": metadata or {}
        }
        
        data_b = {
            "id": id,
            "source_id": source_id or id,
            "application_vector": vectors.get("application_vector", []),
            "innovation_vector": vectors.get("innovation_vector", []),
            "extracted_application": extracted_texts.get("application", "")[:500],
            "extracted_innovation": extracted_texts.get("innovation", "")[:500],
            "region": region or "",
            "maturity": maturity or "",
            "metadata": metadata or {}
        }
        
        coll_a.insert([data_a])
        coll_b.insert([data_b])
        coll_a.flush()
        coll_b.flush()
        g_logger.info(f"Inserted document {id} into {resource_type} (v2)")
        return True
    
    async def insert_batch_v2(
        self,
        resource_type: str,
        items: List[Dict[str, Any]],
        skip_duplicate_check: bool = False
    ) -> int:
        if not items:
            return 0
        
        coll_a, coll_b = self.get_or_create_collections_v2(resource_type)
        
        unique_items = {}
        for item in items:
            item_id = item["id"]
            if item_id not in unique_items:
                unique_items[item_id] = item
        
        if len(unique_items) < len(items):
            g_logger.warning(f"Found {len(items) - len(unique_items)} duplicate IDs in batch, removed")
        
        if not skip_duplicate_check:
            existing_ids = set()
            ids_to_check = list(unique_items.keys())
            
            for i in range(0, len(ids_to_check), 100):
                batch_ids = ids_to_check[i:i+100]
                id_list = ", ".join([f"'{id}'" for id in batch_ids])
                expr = f"id in [{id_list}]"
                
                existing = coll_a.query(
                    expr=expr,
                    output_fields=["id"],
                    limit=len(batch_ids)
                )
                
                for item in existing:
                    existing_ids.add(item["id"])
            
            if existing_ids:
                g_logger.warning(f"Found {len(existing_ids)} existing IDs in database, skipping")
                unique_items = {k: v for k, v in unique_items.items() if k not in existing_ids}
        
        if not unique_items:
            return 0
        
        data_list_a = []
        data_list_b = []
        for item in unique_items.values():
            vectors = item.get("vectors", {})
            extracted = item.get("extracted_texts", {})
            base_metadata = item.get("metadata", {})
            
            data_a = {
                "id": item["id"],
                "source_id": item.get("source_id", item["id"]),
                "domain_vector": vectors.get("domain_vector", []),
                "method_vector": vectors.get("method_vector", []),
                "extracted_domain": extracted.get("domain", "")[:500],
                "extracted_method": extracted.get("method", "")[:500],
                "region": item.get("region", ""),
                "maturity": item.get("maturity", ""),
                "metadata": base_metadata
            }
            
            data_b = {
                "id": item["id"],
                "source_id": item.get("source_id", item["id"]),
                "application_vector": vectors.get("application_vector", []),
                "innovation_vector": vectors.get("innovation_vector", []),
                "extracted_application": extracted.get("application", "")[:500],
                "extracted_innovation": extracted.get("innovation", "")[:500],
                "region": item.get("region", ""),
                "maturity": item.get("maturity", ""),
                "metadata": base_metadata
            }
            
            data_list_a.append(data_a)
            data_list_b.append(data_b)
        
        coll_a.insert(data_list_a)
        coll_b.insert(data_list_b)
        coll_a.flush()
        coll_b.flush()
        
        inserted_count = len(data_list_a)
        g_logger.info(f"Batch inserted {inserted_count} documents into {resource_type} (v2)")
        return inserted_count
    
    async def insert_batch(
        self,
        resource_type: str,
        items: List[Dict[str, Any]],
        skip_duplicate_check: bool = False
    ) -> int:
        """批量插入v1数据（混合检索：dense_vector + sparse_vector）"""
        if not items:
            return 0
        
        collection = self.get_or_create_collection(resource_type)
        
        unique_items = {}
        for item in items:
            item_id = item["id"]
            if item_id not in unique_items:
                unique_items[item_id] = item
        
        if len(unique_items) < len(items):
            g_logger.warning(f"Found {len(items) - len(unique_items)} duplicate IDs in batch, removed")
        
        if not skip_duplicate_check:
            existing_ids = set()
            ids_to_check = list(unique_items.keys())
            
            for i in range(0, len(ids_to_check), 100):
                batch_ids = ids_to_check[i:i+100]
                id_list = ", ".join([f"'{id}'" for id in batch_ids])
                expr = f"id in [{id_list}]"
                
                existing = collection.query(
                    expr=expr,
                    output_fields=["id"],
                    limit=len(batch_ids)
                )
                
                for item in existing:
                    existing_ids.add(item["id"])
            
            if existing_ids:
                g_logger.warning(f"Found {len(existing_ids)} existing IDs in database, skipping")
                unique_items = {k: v for k, v in unique_items.items() if k not in existing_ids}
        
        if not unique_items:
            return 0
        
        data_list = []
        for item in unique_items.values():
            data_list.append({
                "id": item["id"],
                "dense_vector": item.get("dense_vector", []),
                "sparse_vector": item.get("sparse_vector", {}),
                "source_id": item.get("source_id", item["id"]),
                "region": item.get("region", ""),
                "maturity": item.get("maturity", ""),
                "metadata": item.get("metadata", {})
            })
        
        collection.insert(data_list)
        collection.flush()
        
        inserted_count = len(data_list)
        g_logger.info(f"Batch inserted {inserted_count} documents into {resource_type}")
        return inserted_count
    
    async def search_v2(
        self,
        resource_type: str,
        dense_vector: List[float],
        sparse_vector: Optional[Dict[int, float]] = None,
        filter_expr: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        v1_candidates = await self.hybrid_search(
            resource_type=resource_type,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            filter_expr=filter_expr,
            top_k=top_k * 3,
            use_hybrid=use_hybrid
        )
        
        if not v1_candidates:
            return []
        
        coll_a, coll_b = self.get_or_create_collections_v2(resource_type)
        
        ids_to_query = [c["id"] for c in v1_candidates]
        
        v2a_data = {}
        v2b_data = {}
        
        for i in range(0, len(ids_to_query), 100):
            batch_ids = ids_to_query[i:i+100]
            id_list = ", ".join([f"'{id}'" for id in batch_ids])
            expr = f"id in [{id_list}]"
            
            results_a = coll_a.query(
                expr=expr,
                output_fields=["id", "domain_vector", "method_vector", "extracted_domain", "extracted_method"],
                limit=len(batch_ids)
            )
            for r in results_a:
                v2a_data[r["id"]] = r
            
            results_b = coll_b.query(
                expr=expr,
                output_fields=["id", "application_vector", "innovation_vector", "extracted_application", "extracted_innovation"],
                limit=len(batch_ids)
            )
            for r in results_b:
                v2b_data[r["id"]] = r
        
        final_results = []
        for candidate in v1_candidates:
            doc_id = candidate["id"]
            a_data = v2a_data.get(doc_id, {})
            b_data = v2b_data.get(doc_id, {})
            
            final_results.append({
                "id": doc_id,
                "source_id": candidate["source_id"],
                "semantic_score": candidate["score"],
                "domain_vector": a_data.get("domain_vector", []),
                "method_vector": a_data.get("method_vector", []),
                "application_vector": b_data.get("application_vector", []),
                "innovation_vector": b_data.get("innovation_vector", []),
                "extracted_domain": a_data.get("extracted_domain", ""),
                "extracted_method": a_data.get("extracted_method", ""),
                "extracted_application": b_data.get("extracted_application", ""),
                "extracted_innovation": b_data.get("extracted_innovation", ""),
                "region": candidate.get("region"),
                "maturity": candidate.get("maturity"),
                "metadata": candidate.get("metadata")
            })
        
        return final_results
    
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
            seen_ids = set()
            
            for hits in dense_results:
                for hit in hits:
                    doc_id = hit.entity.get("id")
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        combined_scores[doc_id] = {"score": alpha * hit.score, "entity": hit.entity}
            
            for hits in sparse_results:
                for hit in hits:
                    doc_id = hit.entity.get("id")
                    if doc_id in combined_scores:
                        combined_scores[doc_id]["score"] += beta * hit.score
                    elif doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        combined_scores[doc_id] = {"score": beta * hit.score, "entity": hit.entity}
            
            sorted_results = sorted(combined_scores.values(), key=lambda x: x["score"], reverse=True)
            
            final_results = []
            seen_source_ids = set()
            for item in sorted_results:
                entity = item["entity"]
                source_id = entity.get("source_id")
                if source_id not in seen_source_ids:
                    seen_source_ids.add(source_id)
                    final_results.append({
                        "id": entity.get("id"),
                        "source_id": source_id,
                        "score": item["score"],
                        "region": entity.get("region"),
                        "maturity": entity.get("maturity"),
                        "metadata": entity.get("metadata")
                    })
                    if len(final_results) >= top_k:
                        break
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
            seen_source_ids = set()
            for hits in results:
                for hit in hits:
                    entity = hit.entity
                    source_id = entity.get("source_id")
                    if source_id not in seen_source_ids:
                        seen_source_ids.add(source_id)
                        final_results.append({
                            "id": entity.get("id"),
                            "source_id": source_id,
                            "score": hit.score,
                            "region": entity.get("region"),
                            "maturity": entity.get("maturity"),
                            "metadata": entity.get("metadata")
                        })
                        if len(final_results) >= top_k:
                            break
                if len(final_results) >= top_k:
                    break
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
    
    def get_collection_stats(self, resource_type: str) -> Dict[str, Any]:
        collection_name = f"ime_{resource_type}"
        
        self.connect()
        
        if not utility.has_collection(collection_name):
            return {
                "exists": False,
                "count": 0,
                "indexes": []
            }
        
        collection = Collection(collection_name)
        collection.load()
        
        count = collection.num_entities
        
        indexes = []
        for index in collection.indexes:
            indexes.append({
                "field_name": index.field_name,
                "index_type": index.params.get("index_type", "unknown")
            })
        
        return {
            "exists": True,
            "collection_name": collection_name,
            "count": count,
            "indexes": indexes
        }
    
    def list_collections(self) -> List[str]:
        self.connect()
        collections = utility.list_collections()
        ime_collections = [c for c in collections if c.startswith("ime_")]
        return ime_collections

g_milvus = MilvusClient()

__all__ = ["g_milvus", "MilvusClient"]
