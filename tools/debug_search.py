import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
from pymilvus import connections, Collection, utility
from api.core.embedding import g_embedding
from config.app_config import MILVUS_HOST, MILVUS_PORT

def connect_milvus():
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
    except Exception as e:
        pass

async def debug_search(resource_type: str, query: str, top_k: int = 10):
    connect_milvus()
    
    collection_name = f"ime_{resource_type}"
    if not utility.has_collection(collection_name):
        print(f"Collection {collection_name} not exists")
        return
    
    collection = Collection(collection_name)
    collection.load()
    
    print(f"\n查询: {query}")
    print(f"资源类型: {resource_type}")
    print(f"Top K: {top_k}")
    print("=" * 60)
    
    dense_vector, sparse_vector = await g_embedding.get_embedding(query, use_cache=False)
    
    print(f"\n向量维度: {len(dense_vector)}")
    print(f"稀疏向量: {len(sparse_vector) if sparse_vector else 0} 个非零值")
    
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": 100}
    }
    output_fields = ["id", "source_id", "metadata"]
    
    print(f"\n{'=' * 60}")
    print("Dense 向量搜索结果:")
    print(f"{'=' * 60}")
    dense_results = collection.search(
        data=[dense_vector],
        anns_field="dense_vector",
        param=search_params,
        limit=top_k * 2,
        output_fields=output_fields
    )
    
    dense_ids = []
    for hits in dense_results:
        for i, hit in enumerate(hits):
            doc_id = hit.entity.get("id")
            dense_ids.append(doc_id)
            metadata = hit.entity.get("metadata") or {}
            title = metadata.get("title", "")[:30] if isinstance(metadata, dict) else str(metadata)[:30]
            print(f"{i+1:2d}. ID: {doc_id} | Score: {hit.score:.4f} | Title: {title}")
    
    print(f"\nDense结果中的重复ID:")
    from collections import Counter
    dense_counts = Counter(dense_ids)
    for id, count in dense_counts.items():
        if count > 1:
            print(f"  {id}: {count}次")
    if not any(c > 1 for c in dense_counts.values()):
        print("  无重复")
    
    if sparse_vector:
        print(f"\n{'=' * 60}")
        print("Sparse 向量搜索结果:")
        print(f"{'=' * 60}")
        sparse_search_params = {
            "metric_type": "IP",
            "params": {}
        }
        sparse_results = collection.search(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param=sparse_search_params,
            limit=top_k * 2,
            output_fields=output_fields
        )
        
        sparse_ids = []
        for hits in sparse_results:
            for i, hit in enumerate(hits):
                doc_id = hit.entity.get("id")
                sparse_ids.append(doc_id)
                metadata = hit.entity.get("metadata") or {}
                title = metadata.get("title", "")[:30] if isinstance(metadata, dict) else str(metadata)[:30]
                print(f"{i+1:2d}. ID: {doc_id} | Score: {hit.score:.4f} | Title: {title}")
        
        print(f"\nSparse结果中的重复ID:")
        sparse_counts = Counter(sparse_ids)
        for id, count in sparse_counts.items():
            if count > 1:
                print(f"  {id}: {count}次")
        if not any(c > 1 for c in sparse_counts.values()):
            print("  无重复")
        
        print(f"\n{'=' * 60}")
        print("两个结果集中都出现的ID:")
        common = set(dense_ids) & set(sparse_ids)
        for id in common:
            print(f"  {id}")
        if not common:
            print("  无交集")

async def main():
    resource_type = input("资源类型 (默认: tec): ").strip() or "tec"
    query = input("查询词 (默认: 纳米材料涂层): ").strip() or "纳米材料涂层"
    top_k = int(input("Top K (默认: 10): ").strip() or "10")
    
    await debug_search(resource_type, query, top_k)

if __name__ == "__main__":
    asyncio.run(main())
