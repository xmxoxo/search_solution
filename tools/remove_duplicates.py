import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pymilvus import connections, Collection, utility
from config.app_config import MILVUS_HOST, MILVUS_PORT, RESOURCE_TYPES

def connect_milvus():
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
    except Exception as e:
        pass

def get_duplicate_source_ids(collection_name: str) -> dict:
    """获取重复的source_id及其对应的所有id"""
    connect_milvus()
    
    if not utility.has_collection(collection_name):
        return {}
    
    collection = Collection(collection_name)
    collection.load()
    
    source_id_to_ids = {}
    page_size = 500
    last_id = ""
    total = 0
    
    print("正在扫描数据...")
    while True:
        expr = f"id > '{last_id}'" if last_id else "id != ''"
        
        results = collection.query(
            expr=expr,
            output_fields=["id", "source_id"],
            limit=page_size
        )
        
        if not results:
            break
        
        for item in results:
            total += 1
            source_id = item.get("source_id", "")
            doc_id = item.get("id", "")
            
            if source_id not in source_id_to_ids:
                source_id_to_ids[source_id] = []
            source_id_to_ids[source_id].append(doc_id)
            
            last_id = doc_id
        
        if len(results) < page_size:
            break
        
        if total % 5000 == 0:
            print(f"  已扫描 {total} 条")
    
    print(f"扫描完成，共 {total} 条数据")
    
    duplicates = {}
    for source_id, ids in source_id_to_ids.items():
        if len(ids) > 1:
            duplicates[source_id] = ids
    
    return duplicates

def delete_duplicates(collection_name: str, duplicates: dict):
    """删除source_id重复的数据，每个source_id只保留第一条"""
    connect_milvus()
    collection = Collection(collection_name)
    
    total_deleted = 0
    source_id_list = list(duplicates.items())
    
    for i, (source_id, ids) in enumerate(source_id_list):
        ids_to_delete = ids[1:]
        
        if ids_to_delete:
            id_list = ", ".join([f"'{id}'" for id in ids_to_delete])
            expr = f"id in [{id_list}]"
            collection.delete(expr)
            collection.flush()
            total_deleted += len(ids_to_delete)
            
            print(f"  [{i+1}/{len(duplicates)}] source_id: '{source_id[:40]}...', 保留1条, 删除{len(ids_to_delete)}条")
    
    return total_deleted

def main():
    print("=" * 60)
    print("source_id重复数据清理工具")
    print("=" * 60)
    
    resource_type = input("请输入资源类型 (tec/expert/project等): ").strip()
    
    if resource_type not in RESOURCE_TYPES:
        print(f"错误: 无效的资源类型 '{resource_type}'")
        print(f"有效类型: {', '.join(RESOURCE_TYPES)}")
        return
    
    collection_name = f"ime_{resource_type}"
    
    print(f"\n正在检查 {collection_name} 中的source_id重复数据...")
    duplicates = get_duplicate_source_ids(collection_name)
    
    if not duplicates:
        print("未发现source_id重复的数据")
        return
    
    total_duplicate_groups = len(duplicates)
    total_duplicate_records = sum(len(ids) - 1 for ids in duplicates.values())
    
    print(f"\n发现 {total_duplicate_groups} 个重复的source_id，涉及 {total_duplicate_records} 条重复记录")
    print(f"\n前10组重复source_id预览:")
    for i, (source_id, ids) in enumerate(list(duplicates.items())[:10]):
        print(f"  {i+1}. '{source_id[:50]}...' ({len(ids)}条)")
    
    confirm = input(f"\n确认删除这些重复数据? 将删除 {total_duplicate_records} 条，保留 {total_duplicate_groups} 条 (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    print("\n开始删除重复数据...")
    deleted = delete_duplicates(collection_name, duplicates)
    
    print(f"\n完成! 共删除 {deleted} 条source_id重复的数据")

if __name__ == "__main__":
    main()
