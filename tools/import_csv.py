import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import csv
import json
import argparse
from tqdm import tqdm
from pymilvus import connections, Collection
from api.core.embedding import g_embedding
from api.core.milvus_client import g_milvus
from api.utils.logger import g_logger
from config.app_config import RESOURCE_TYPES, MILVUS_HOST, MILVUS_PORT

MAX_RETRIES = 3
RETRY_DELAY = 5

def connect_milvus():
    """连接Milvus"""
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
    except Exception as e:
        pass

def get_existing_ids(collection_name: str) -> set:
    """获取Collection中已存在的ID集合（使用游标分页）"""
    try:
        connect_milvus()
        from pymilvus import utility
        
        if not utility.has_collection(collection_name):
            return set()
        
        collection = Collection(collection_name)
        collection.load()
        
        existing_ids = set()
        page_size = 1000
        last_id = ""
        
        while True:
            expr = f"id > '{last_id}'" if last_id else "id != ''"
            
            results = collection.query(
                expr=expr,
                output_fields=["id"],
                limit=page_size
            )
            
            if not results:
                break
            
            for item in results:
                existing_ids.add(item["id"])
                last_id = item["id"]
            
            if len(results) < page_size:
                break
        
        g_logger.info(f"Found {len(existing_ids)} existing IDs in {collection_name}")
        return existing_ids
    except Exception as e:
        g_logger.warning(f"Failed to get existing IDs: {e}")
        return set()

async def get_embeddings_with_retry(texts: list, max_retries: int = MAX_RETRIES) -> list:
    """批量向量化，带重试机制"""
    for attempt in range(max_retries):
        try:
            embeddings = await g_embedding.get_embeddings_batch(
                texts,
                use_cache=False
            )
            return embeddings
            
        except Exception as e:
            if attempt < max_retries - 1:
                g_logger.warning(f"Batch embedding retry {attempt+1}/{max_retries}: {e}")
                await asyncio.sleep(RETRY_DELAY)
            else:
                g_logger.error(f"Batch embedding failed after {max_retries} retries: {e}")
                raise

async def process_batch(batch_data: list, resource_type: str) -> tuple:
    """处理一批数据（批量向量化 + 批量插入，带重试）"""
    if not batch_data:
        return 0, 0, []
    
    texts = []
    valid_items = []
    
    for item in batch_data:
        item_id = item.get('id', '').strip()
        title = item.get('title', '').strip()
        body = item.get('body', '').strip()
        
        if not item_id or not body:
            continue
        
        raw_text = f"{title} {body}" if title else body
        texts.append(raw_text)
        valid_items.append({
            'id': item_id,
            'title': title,
            'body': body
        })
    
    if not texts:
        return 0, len(batch_data), [item.get('id') for item in batch_data]
    
    try:
        embeddings = await get_embeddings_with_retry(texts)
        
        insert_items = []
        for i, item in enumerate(valid_items):
            dense_vector, sparse_vector = embeddings[i]
            insert_items.append({
                'id': item['id'],
                'dense_vector': dense_vector,
                'sparse_vector': sparse_vector,
                'source_id': item['id'],
                'region': '',
                'maturity': '',
                'metadata': {
                    'title': item['title'],
                    'body': item['body']
                }
            })
        
        inserted_count = await g_milvus.insert_batch(
            resource_type=resource_type,
            items=insert_items,
            skip_duplicate_check=True
        )
        
        failed_count = len(batch_data) - inserted_count
        failed_ids = []
        if failed_count > 0:
            all_ids = set(item.get('id', '') for item in batch_data)
            success_ids = set(item['id'] for item in insert_items)
            failed_ids = list(all_ids - success_ids)
        
        return inserted_count, failed_count, failed_ids
        
    except Exception as e:
        g_logger.error(f"Batch processing error: {e}")
        return 0, len(batch_data), [item.get('id', f'unknown_{i}') for i, item in enumerate(batch_data)]

async def import_from_csv(
    csv_file: str,
    resource_type: str,
    batch_size: int = 100,
    dry_run: bool = False,
    skip_existing: bool = True
):
    """从CSV文件批量导入数据"""
    
    if resource_type not in RESOURCE_TYPES:
        print(f"错误: 资源类型 '{resource_type}' 无效")
        print(f"有效类型: {', '.join(RESOURCE_TYPES)}")
        return
    
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"错误: 文件不存在 - {csv_file}")
        return
    
    print(f"=" * 60)
    print(f"数据导入工具")
    print(f"=" * 60)
    print(f"CSV文件: {csv_file}")
    print(f"资源类型: {resource_type}")
    print(f"批次大小: {batch_size}")
    print(f"试运行: {'是' if dry_run else '否'}")
    print(f"跳过已存在: {'是' if skip_existing else '否'}")
    print(f"=" * 60)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    total_rows = len(rows)
    print(f"共发现 {total_rows} 条数据")
    
    if total_rows == 0:
        print("CSV文件为空，无需导入")
        return
    
    print(f"\n字段名: {list(rows[0].keys())}")
    print(f"\n第一条数据预览:")
    for k, v in list(rows[0].items())[:3]:
        print(f"  {k}: {v[:50] if len(v) > 50 else v}")
    
    collection_name = f"ime_{resource_type}"
    
    existing_ids = set()
    if skip_existing:
        print(f"\n正在查询已存在的ID...")
        existing_ids = get_existing_ids(collection_name)
        print(f"已存在 {len(existing_ids)} 条数据")
    
    rows_to_import = []
    skipped_count = 0
    
    for row in rows:
        item_id = row.get('id', '').strip()
        if skip_existing and item_id in existing_ids:
            skipped_count += 1
            continue
        rows_to_import.append(row)
    
    if skipped_count > 0:
        print(f"跳过已存在的数据: {skipped_count} 条")
    
    if len(rows_to_import) == 0:
        print("没有需要导入的数据")
        return
    
    print(f"待导入数据: {len(rows_to_import)} 条")
    
    confirm = input("\n确认开始导入? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    total_success = 0
    total_failed = 0
    all_failed_ids = []
    
    print(f"\n开始导入...")
    
    for i in tqdm(range(0, len(rows_to_import), batch_size), desc="导入进度"):
        batch = rows_to_import[i:i+batch_size]
        
        if dry_run:
            total_success += len(batch)
            continue
        
        success, failed, failed_ids = await process_batch(batch, resource_type)
        total_success += success
        total_failed += failed
        all_failed_ids.extend(failed_ids)
    
    print(f"\n{'=' * 60}")
    print(f"导入完成!")
    print(f"{'=' * 60}")
    print(f"总计: {total_rows} 条")
    print(f"已存在: {skipped_count} 条（已跳过）")
    print(f"成功: {total_success} 条")
    print(f"失败: {total_failed} 条")
    
    if all_failed_ids:
        print(f"\n失败的ID列表:")
        for fid in all_failed_ids[:20]:
            print(f"  - {fid}")
        if len(all_failed_ids) > 20:
            print(f"  ... 还有 {len(all_failed_ids) - 20} 条")

def main():
    parser = argparse.ArgumentParser(
        description='从CSV文件批量导入数据到向量数据库',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python tools/import_csv.py --csv F:/data/tec_body.csv --resource tec
  python tools/import_csv.py --csv F:/data/tec_body.csv --resource tec --dry-run
  python tools/import_csv.py --csv F:/data/tec_body.csv --resource tec --batch-size 200
  python tools/import_csv.py --csv F:/data/tec_body.csv --resource tec --no-skip-existing
        '''
    )
    
    parser.add_argument(
        '--csv',
        required=True,
        help='CSV文件路径（必须包含 id, title, body 字段）'
    )
    
    parser.add_argument(
        '--resource',
        required=True,
        choices=RESOURCE_TYPES,
        help='资源类型'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='批次大小（默认: 100）'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='试运行模式，不实际写入数据'
    )
    
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='不跳过已存在的ID'
    )
    
    args = parser.parse_args()
    
    asyncio.run(import_from_csv(
        csv_file=args.csv,
        resource_type=args.resource,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        skip_existing=not args.no_skip_existing
    ))

if __name__ == "__main__":
    main()
