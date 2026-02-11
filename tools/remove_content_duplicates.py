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

def get_content_duplicates(collection_name: str) -> dict:
    """按内容分组，找出内容重复的记录"""
    connect_milvus()
    
    if not utility.has_collection(collection_name):
        return {}
    
    collection = Collection(collection_name)
    collection.load()
    
    content_to_ids = {}
    page_size = 500
    last_id = ""
    total = 0
    
    print("正在扫描数据...")
    while True:
        expr = f"id > '{last_id}'" if last_id else "id != ''"
        
        results = collection.query(
            expr=expr,
            output_fields=["id", "metadata"],
            limit=page_size
        )
        
        if not results:
            break
        
        for item in results:
            total += 1
            metadata = item.get("metadata") or {}
            title = metadata.get("title", "") if isinstance(metadata, dict) else ""
            body = metadata.get("body", "") if isinstance(metadata, dict) else ""
            
            content_key = f"{title}|||{body}"
            
            if content_key not in content_to_ids:
                content_to_ids[content_key] = []
            content_to_ids[content_key].append(item["id"])
            
            last_id = item["id"]
        
        if len(results) < page_size:
            break
        
        if total % 5000 == 0:
            print(f"  已扫描 {total} 条")
    
    print(f"扫描完成，共 {total} 条数据")
    
    duplicates = {}
    for content, ids in content_to_ids.items():
        if len(ids) > 1:
            duplicates[content] = ids
    
    return duplicates

def delete_content_duplicates(collection_name: str, duplicates: dict):
    """删除内容重复的数据，只保留第一条"""
    connect_milvus()
    collection = Collection(collection_name)
    
    total_deleted = 0
    content_list = list(duplicates.items())
    
    for i, (content, ids) in enumerate(content_list):
        ids_to_delete = ids[1:]
        
        if ids_to_delete:
            id_list = ", ".join([f"'{id}'" for id in ids_to_delete])
            expr = f"id in [{id_list}]"
            collection.delete(expr)
            collection.flush()
            total_deleted += len(ids_to_delete)
            
            title = content.split("|||")[0][:30]
            print(f"  [{i+1}/{len(duplicates)}] 删除重复: '{title}...', 保留1条, 删除{len(ids_to_delete)}条")
    
    return total_deleted

def main():
    print("=" * 60)
    print("内容重复数据清理工具")
    print("=" * 60)
    
    resource_type = input("请输入资源类型 (tec/expert/project等): ").strip()
    
    if resource_type not in RESOURCE_TYPES:
        print(f"错误: 无效的资源类型 '{resource_type}'")
        print(f"有效类型: {', '.join(RESOURCE_TYPES)}")
        return
    
    collection_name = f"ime_{resource_type}"
    
    print(f"\n正在检查 {collection_name} 中的内容重复数据...")
    duplicates = get_content_duplicates(collection_name)
    
    if not duplicates:
        print("未发现内容重复的数据")
        return
    
    total_duplicate_groups = len(duplicates)
    total_duplicate_records = sum(len(ids) - 1 for ids in duplicates.values())
    
    print(f"\n发现 {total_duplicate_groups} 组内容重复的数据，涉及 {total_duplicate_records} 条重复记录")
    print(f"\n前10组重复内容预览:")
    for i, (content, ids) in enumerate(list(duplicates.items())[:10]):
        title = content.split("|||")[0][:40]
        print(f"  {i+1}. '{title}...' ({len(ids)}条)")
    
    confirm = input(f"\n确认删除这些重复数据? 将删除 {total_duplicate_records} 条，保留 {total_duplicate_groups} 条 (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    print("\n开始删除重复数据...")
    deleted = delete_content_duplicates(collection_name, duplicates)
    
    print(f"\n完成! 共删除 {deleted} 条内容重复的数据")

if __name__ == "__main__":
    main()
