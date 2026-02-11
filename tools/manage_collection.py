import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pymilvus import connections, utility, Collection
from config.app_config import MILVUS_HOST, MILVUS_PORT, RESOURCE_TYPES

def connect_milvus():
    """连接Milvus"""
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        print(f"✅ 已连接到 Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
        return True
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

def disconnect_milvus():
    """断开Milvus连接"""
    connections.disconnect("default")
    print("🔌 已断开连接")

def list_collections():
    """列出所有Collection"""
    collections = utility.list_collections()
    print(f"\n📚 当前Collection列表 ({len(collections)}):")
    if collections:
        for col in collections:
            print(f"  - {col}")
    else:
        print("  (空)")
    return collections

def count_collection(collection_name: str) -> int:
    """统计指定Collection的数据量"""
    try:
        if not utility.has_collection(collection_name):
            return 0
        
        collection = Collection(collection_name)
        collection.load()
        return collection.num_entities
    except Exception as e:
        print(f"  ⚠️  统计 {collection_name} 失败: {e}")
        return 0

def count_all_resources():
    """统计所有资源类型的数据量"""
    print(f"\n📊 资源数据统计:")
    print("-" * 40)
    
    total = 0
    stats = []
    
    for resource_type in RESOURCE_TYPES:
        collection_name = f"ime_{resource_type}"
        count = count_collection(collection_name)
        stats.append((resource_type, collection_name, count))
        total += count
    
    for resource_type, collection_name, count in stats:
        print(f"  {resource_type:12} -> {collection_name:15} : {count:>8} 条")
    
    print("-" * 40)
    print(f"  {'总计':<12} -> {'':15} : {total:>8} 条")
    print("-" * 40)

def delete_collection(collection_name: str):
    """删除指定Collection"""
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"🗑️  已删除 Collection: {collection_name}")
    else:
        print(f"⚠️  Collection不存在: {collection_name}")

def delete_all_ime_collections():
    """删除所有IME相关的Collection"""
    collections = utility.list_collections()
    ime_collections = [c for c in collections if c.startswith("ime_")]
    
    if not ime_collections:
        print("没有找到IME相关的Collection")
        return
    
    print(f"\n找到以下IME Collection ({len(ime_collections)}):")
    for col in ime_collections:
        print(f"  - {col}")
    
    confirm = input("\n确认删除所有这些Collection? (y/n): ").strip().lower()
    if confirm == 'y':
        for col in ime_collections:
            utility.drop_collection(col)
            print(f"🗑️  已删除: {col}")
        print("\n✅ 所有IME Collection已删除")
    else:
        print("已取消")

def main():
    parser = argparse.ArgumentParser(
        description='Milvus Collection管理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python tools/manage_collection.py --list
  python tools/manage_collection.py --count
  python tools/manage_collection.py --delete ime_tec
  python tools/manage_collection.py --delete-all
        '''
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有Collection'
    )
    
    parser.add_argument(
        '--count',
        action='store_true',
        help='统计所有资源类型的数据量'
    )
    
    parser.add_argument(
        '--delete',
        type=str,
        help='删除指定的Collection'
    )
    
    parser.add_argument(
        '--delete-all',
        action='store_true',
        help='删除所有IME相关的Collection'
    )
    
    args = parser.parse_args()
    
    if not connect_milvus():
        return
    
    try:
        if args.list:
            list_collections()
        elif args.count:
            count_all_resources()
        elif args.delete:
            delete_collection(args.delete)
        elif args.delete_all:
            delete_all_ime_collections()
        else:
            parser.print_help()
    finally:
        disconnect_milvus()

if __name__ == "__main__":
    main()
