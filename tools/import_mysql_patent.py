import sys
import os
import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 设置默认特征提取模型使用 qwen3:32b qwen-flash
os.environ['LLM_MODEL_NAME'] = 'qwen-flash'

import asyncio
import argparse
import pymysql
import traceback
from tqdm import tqdm
from api.core.feature_extractor import g_feature_extractor
from api.core.milvus_client import g_milvus
from api.core.embedding import g_embedding
from api.utils.logger import g_logger
from config.app_config import DATABASE_CONNECT_STRING

BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 5
MAX_ROWS_PER_FILE = 10000
DATA_DIR = PROJECT_ROOT / "data"
LAST_ID_FILE = DATA_DIR / "last_patent_id.txt"

def save_last_successful_id(last_id: str, key=""):
    """保存最后成功的ID到文件"""
    try:
        DATA_DIR.mkdir(exist_ok=True)
        if key:
            fname = DATA_DIR / f"last_patent_id_{key}.txt"
        else:
            fname = LAST_ID_FILE

        with open(fname, 'w', encoding='utf-8') as f:
            f.write(last_id)
        g_logger.info(f"Saved last successful ID: {last_id}")
    except Exception as e:
        g_logger.error(f"Failed to save last successful ID: {e}")

def get_last_successful_id(key="") -> str:
    """获取最后成功的ID"""
    try:
        if key:
            fname = DATA_DIR / f"last_patent_id_{key}.txt"
        else:
            fname = LAST_ID_FILE

        if fname.exists():
            with open(fname, 'r', encoding='utf-8') as f:
                last_id = f.read().strip()
            g_logger.info(f"Loaded last successful ID: {last_id}")
            return last_id
        return ""
    except Exception as e:
        g_logger.error(f"Failed to load last successful ID: {e}")
        return ""

def save_intermediate_data(items: list, begin_id: str, resource_type: str) -> str:
    """保存中间数据到CSV文件（包含v1和v2的向量）"""
    DATA_DIR.mkdir(exist_ok=True)
    filename = f"patent_{begin_id}_{resource_type}.csv"
    filepath = DATA_DIR / filename
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'id', 'source_id', 'region', 'maturity',
                'dense_vector', 'sparse_vector',
                'domain_vector', 'method_vector', 'application_vector', 'innovation_vector',
                'extracted_domain', 'extracted_method', 'extracted_application', 'extracted_innovation'
            ])
            
            for item in items:
                # 处理 sparse_vector 格式
                sparse_vector = item.get('sparse_vector', {})
                if sparse_vector is None:
                    sparse_vector = {}
                elif hasattr(sparse_vector, 'indices') and hasattr(sparse_vector, 'values'):
                    sparse_vector = dict(zip(sparse_vector.indices, sparse_vector.values))
                elif not isinstance(sparse_vector, dict):
                    sparse_vector = {}
                
                writer.writerow([
                    item['id'],
                    item['source_id'],
                    item['region'],
                    item['maturity'],
                    json.dumps(item.get('dense_vector', [])),
                    json.dumps(sparse_vector),
                    json.dumps(item['vectors']['domain_vector']),
                    json.dumps(item['vectors']['method_vector']),
                    json.dumps(item['vectors']['application_vector']),
                    json.dumps(item['vectors']['innovation_vector']),
                    item['extracted_texts'].get('domain', ''),
                    item['extracted_texts'].get('method', ''),
                    item['extracted_texts'].get('application', ''),
                    item['extracted_texts'].get('innovation', '')
                ])
        
        g_logger.info(f"Saved {len(items)} items to {filepath}")
        return str(filepath)
    except Exception as e:
        g_logger.error(f"Failed to save intermediate data: {e}")
        g_logger.error(traceback.format_exc())
        return ""

def load_intermediate_data(filepath: str) -> list:
    """从CSV文件加载中间数据（包含v1和v2的向量）"""
    items = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append({
                    'id': row['id'],
                    'source_id': row['source_id'],
                    'region': row['region'],
                    'maturity': row['maturity'],
                    'dense_vector': json.loads(row['dense_vector']),
                    'sparse_vector': json.loads(row['sparse_vector']),
                    'vectors': {
                        'domain_vector': json.loads(row['domain_vector']),
                        'method_vector': json.loads(row['method_vector']),
                        'application_vector': json.loads(row['application_vector']),
                        'innovation_vector': json.loads(row['innovation_vector'])
                    },
                    'extracted_texts': {
                        'domain': row['extracted_domain'],
                        'method': row['extracted_method'],
                        'application': row['extracted_application'],
                        'innovation': row['extracted_innovation']
                    }
                })
        g_logger.info(f"Loaded {len(items)} items from {filepath}")
        return items
    except Exception as e:
        g_logger.error(f"Failed to load intermediate data: {e}")
        g_logger.error(traceback.format_exc())
        return []

async def get_total_count(connection, start_id: str = None) -> int:
    """获取符合条件的专利总数"""
    try:
        with connection.cursor() as cursor:
            if start_id:
                sql = """
                SELECT COUNT(*)
                FROM t_patent_trade_info
                WHERE patent_type IS NOT NULL AND patent_name IS NOT NULL AND descript IS NOT NULL
                AND id > %s
                """
                g_logger.info(f"Executing count SQL with start_id={start_id}: {sql.strip()}")
                cursor.execute(sql, (start_id,))
            else:
                sql = """
                SELECT COUNT(*)
                FROM t_patent_trade_info
                WHERE patent_type IS NOT NULL AND patent_name IS NOT NULL AND descript IS NOT NULL
                """
                g_logger.info(f"Executing count SQL: {sql.strip()}")
                cursor.execute(sql)
            
            result = cursor.fetchone()
            g_logger.info(f"Count result: {result}")
            if result:
                # 处理字典类型的结果
                if isinstance(result, dict):
                    count = result.get('COUNT(*)', 0)
                else:
                    count = result[0]
                g_logger.info(f"Total count: {count}")
                return count
            else:
                g_logger.warning("No result returned from count query")
                return 0
    except Exception as e:
        g_logger.error(f"Failed to get total count: {e}")
        g_logger.error(traceback.format_exc())
        return 0

async def fetch_patent_batch(connection, offset: int, limit: int, start_id: str = None) -> list:
    """批量获取专利数据"""
    try:
        with connection.cursor() as cursor:
            if start_id:
                sql = """
                SELECT id, patent_name as title, descript as body
                FROM t_patent_trade_info
                WHERE patent_type IS NOT NULL AND patent_name IS NOT NULL AND descript IS NOT NULL
                AND id > %s
                ORDER BY id
                LIMIT %s OFFSET %s
                """
                g_logger.info(f"Executing fetch SQL with start_id={start_id}, limit={limit}, offset={offset}")
                cursor.execute(sql, (start_id, limit, offset))
            else:
                sql = """
                SELECT id, patent_name as title, descript as body
                FROM t_patent_trade_info
                WHERE patent_type IS NOT NULL AND patent_name IS NOT NULL AND descript IS NOT NULL
                ORDER BY id
                LIMIT %s OFFSET %s
                """
                g_logger.info(f"Executing fetch SQL with limit={limit}, offset={offset}")
                cursor.execute(sql, (limit, offset))
            
            results = cursor.fetchall()
            g_logger.info(f"Fetch results count: {len(results)}")
            
            patents = []
            for row in results:
                try:
                    # 处理字典类型的结果
                    if isinstance(row, dict):
                        patent = {
                            "id": str(row.get('id', '')),
                            "title": row.get('title', ''),
                            "body": row.get('body', '')
                        }
                    else:
                        patent = {
                            "id": str(row[0]),
                            "title": row[1],
                            "body": row[2]
                        }
                    patents.append(patent)
                except Exception as row_e:
                    g_logger.error(f"Failed to process row: {row_e}")
                    continue
            g_logger.info(f"Processed patents count: {len(patents)}")
            return patents
    except Exception as e:
        g_logger.error(f"Failed to fetch patent batch: {e}")
        g_logger.error(traceback.format_exc())
        return []

async def process_batch(patents: list, resource_type: str, failed_ids_file: str) -> tuple:
    """处理一批专利数据，同时导入v1和v2"""
    if not patents:
        return 0, 0, []
    
    failed_ids = []
    v1_insert_items = []
    v2_insert_items = []
    
    for patent in patents:
        try:
            title = patent.get("title", "")
            body = patent.get("body", "")
            
            if not body:
                failed_ids.append(patent["id"])
                continue
            
            # 生成v1混合检索向量
            full_text = f"{title} {body}"
            dense_vector, sparse_vector = await g_embedding.get_embedding(full_text, use_cache=False)
            
            # 处理 sparse_vector 格式
            if sparse_vector is None:
                sparse_vector = {}
            elif hasattr(sparse_vector, 'indices') and hasattr(sparse_vector, 'values'):
                # 转换为字典格式
                sparse_vector = dict(zip(sparse_vector.indices, sparse_vector.values))
            elif not isinstance(sparse_vector, dict):
                sparse_vector = {}
            
            # 生成v2扩展维度向量
            features = await g_feature_extractor.extract_all_features(title, body)
            
            # v1数据（混合检索）
            v1_insert_items.append({
                'id': patent['id'],
                'dense_vector': dense_vector,
                'sparse_vector': sparse_vector,
                'source_id': patent['id'],
                'region': '',
                'maturity': '',
                'metadata': {}
            })
            
            # v2数据（扩展维度）
            v2_insert_items.append({
                'id': patent['id'],
                'vectors': {
                    'domain_vector': features['domain_vector'],
                    'method_vector': features['method_vector'],
                    'application_vector': features['application_vector'],
                    'innovation_vector': features['innovation_vector']
                },
                'extracted_texts': features['extracted_texts'],
                'source_id': patent['id'],
                'region': '',
                'maturity': '',
                'metadata': {}
            })
        except Exception as e:
            g_logger.error(f"Failed to process patent {patent['id']}: {e}")
            failed_ids.append(patent['id'])
    
    if not v1_insert_items:
        return 0, len(patents), failed_ids
    
    # 插入v1数据（混合检索）
    for attempt in range(MAX_RETRIES):
        try:
            inserted_count = await g_milvus.insert_batch(
                resource_type=resource_type,
                items=v1_insert_items,
                skip_duplicate_check=True
            )
            
            if inserted_count > 0:
                g_logger.info(f"Inserted {inserted_count} patents into v1 collection")
                break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (attempt + 1)
                g_logger.warning(f"V1 insert retry {attempt+1}/{MAX_RETRIES} after {delay}s: {e}")
                await asyncio.sleep(delay)
            else:
                g_logger.error(f"V1 insert failed after {MAX_RETRIES} attempts: {e}")
                all_failed = [patent['id'] for patent in patents]
                return 0, len(all_failed), all_failed
    
    # 插入v2数据（扩展维度）
    for attempt in range(MAX_RETRIES):
        try:
            inserted_count = await g_milvus.insert_batch_v2(
                resource_type=resource_type,
                items=v2_insert_items,
                skip_duplicate_check=True
            )
            
            if inserted_count > 0:
                g_logger.info(f"Inserted {inserted_count} patents into v2 collections")
                return inserted_count, len(failed_ids), failed_ids
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (attempt + 1)
                g_logger.warning(f"V2 insert retry {attempt+1}/{MAX_RETRIES} after {delay}s: {e}")
                await asyncio.sleep(delay)
            else:
                g_logger.error(f"V2 insert failed after {MAX_RETRIES} attempts: {e}")
                all_failed = [patent['id'] for patent in patents]
                return 0, len(all_failed), all_failed

async def import_from_intermediate_file(filepath: str, resource_type: str, batch_size: int = BATCH_SIZE) -> tuple:
    """从中间数据文件导入到Milvus，同时导入v1和v2"""
    g_logger.info(f"Importing from intermediate file: {filepath}")
    
    items = load_intermediate_data(filepath)
    if not items:
        g_logger.error("No items loaded from intermediate file")
        return 0, 0
    
    total_success = 0
    total_failed = 0
    all_failed_ids = []
    last_successful_id = ""
    
    with tqdm(total=len(items), desc="Importing from intermediate file") as pbar:
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            
            # 准备v1和v2的数据
            v1_insert_items = []
            v2_insert_items = []
            
            for item in batch:
                try:
                    # 处理 sparse_vector 格式
                    sparse_vector = item['sparse_vector']
                    if sparse_vector is None:
                        sparse_vector = {}
                    elif hasattr(sparse_vector, 'indices') and hasattr(sparse_vector, 'values'):
                        sparse_vector = dict(zip(sparse_vector.indices, sparse_vector.values))
                    elif not isinstance(sparse_vector, dict):
                        sparse_vector = {}
                    
                    # v1数据（混合检索）
                    v1_insert_items.append({
                        'id': item['id'],
                        'dense_vector': item['dense_vector'],
                        'sparse_vector': sparse_vector,
                        'source_id': item['source_id'],
                        'region': item['region'],
                        'maturity': item['maturity'],
                        'metadata': {}
                    })
                    
                    # v2数据（扩展维度）
                    v2_insert_items.append({
                        'id': item['id'],
                        'vectors': item['vectors'],
                        'extracted_texts': item['extracted_texts'],
                        'source_id': item['source_id'],
                        'region': item['region'],
                        'maturity': item['maturity'],
                        'metadata': {}
                    })
                except Exception as e:
                    g_logger.error(f"Failed to prepare item {item['id']}: {e}")
                    all_failed_ids.append(item['id'])
            
            # 插入v1数据
            if v1_insert_items:
                for attempt in range(MAX_RETRIES):
                    try:
                        inserted_count = await g_milvus.insert_batch(
                            resource_type=resource_type,
                            items=v1_insert_items,
                            skip_duplicate_check=True
                        )
                        
                        if inserted_count > 0:
                            g_logger.info(f"Inserted {inserted_count} items into v1 collection")
                            break
                    except Exception as e:
                        if attempt < MAX_RETRIES - 1:
                            delay = RETRY_DELAY * (attempt + 1)
                            g_logger.warning(f"V1 insert retry {attempt+1}/{MAX_RETRIES} after {delay}s: {e}")
                            await asyncio.sleep(delay)
                        else:
                            g_logger.error(f"V1 insert failed after {MAX_RETRIES} attempts: {e}")
                            total_failed += len(batch)
                            all_failed_ids.extend([item['id'] for item in batch])
            
            # 插入v2数据
            if v2_insert_items:
                for attempt in range(MAX_RETRIES):
                    try:
                        inserted_count = await g_milvus.insert_batch_v2(
                            resource_type=resource_type,
                            items=v2_insert_items,
                            skip_duplicate_check=True
                        )
                        
                        if inserted_count > 0:
                            total_success += inserted_count
                            # 保存最后成功的ID
                            last_successful_id = batch[-1]['id']
                            save_last_successful_id(last_successful_id)
                            
                            pbar.update(len(batch))
                            pbar.set_postfix({"success": total_success, "failed": total_failed})
                            break
                    except Exception as e:
                        if attempt < MAX_RETRIES - 1:
                            delay = RETRY_DELAY * (attempt + 1)
                            g_logger.warning(f"V2 insert retry {attempt+1}/{MAX_RETRIES} after {delay}s: {e}")
                            await asyncio.sleep(delay)
                        else:
                            g_logger.error(f"V2 insert failed after {MAX_RETRIES} attempts: {e}")
                            total_failed += len(batch)
                            all_failed_ids.extend([item['id'] for item in batch])
    
    g_logger.info(f"Import from intermediate file completed:")
    g_logger.info(f"  Total: {len(items)}")
    g_logger.info(f"  Success: {total_success}")
    g_logger.info(f"  Failed: {total_failed}")
    
    return total_success, total_failed

async def import_mysql_patent(
    resource_type: str,
    batch_size: int = BATCH_SIZE,
    limit: int = None,
    dry_run: bool = False,
    save_intermediate: bool = True,
    start_id: str = None
):
    """从MySQL数据库导入专利数据"""
    g_logger.info(f"Starting MySQL patent import to {resource_type}")
    # 自动读取最新ID
    if start_id is None:
        start_id = get_last_successful_id(key=start_id)
    g_logger.info(f"Batch size: {batch_size}, Limit: {limit}, Save intermediate: {save_intermediate}, Start ID: {start_id}")
    
    failed_ids_file = f"failed_patent_ids_{resource_type}.txt"
    total_success = 0
    total_failed = 0
    all_failed_ids = []
    
    # 用于保存中间数据的缓冲区
    intermediate_buffer = []
    current_file_count = 0
    last_begin_id = ""
    
    try:
        from urllib.parse import urlparse
        
        parsed = urlparse(DATABASE_CONNECT_STRING)
        db_config = {
            "host": parsed.hostname,
            "port": parsed.port or 3306,
            "user": parsed.username,
            "password": parsed.password,
            "database": parsed.path.lstrip("/"),
            "charset": "utf8mb4",
            "cursorclass": pymysql.cursors.DictCursor
        }
        
        connection = pymysql.connect(**db_config)
        g_logger.info(f"Connected to MySQL database: {parsed.hostname}")
        
        total_count = await get_total_count(connection, start_id)
        if limit:
            total_count = min(total_count, limit)
        
        g_logger.info(f"Total patents to import: {total_count}")
        
        if dry_run:
            g_logger.info(f"Dry run mode: would import {total_count} patents")
            return
        
        with tqdm(total=total_count, desc="Importing patents") as pbar:
            offset = 0
            while offset < total_count:
                current_limit = min(batch_size, total_count - offset)
                patents = await fetch_patent_batch(connection, offset, current_limit, start_id)
                
                if not patents:
                    g_logger.warning(f"No patents fetched at offset {offset}")
                    break
                
                # 处理批量数据
                success, failed, failed_ids = await process_batch(
                    patents, resource_type, failed_ids_file
                )
                
                total_success += success
                total_failed += failed
                all_failed_ids.extend(failed_ids)
                
                # 保存最后成功的ID
                if success > 0 and patents:
                    last_successful_id = patents[-1]['id']
                    save_last_successful_id(last_successful_id, key=start_id)
                
                # 保存中间数据
                if save_intermediate:
                    for patent in patents:
                        try:
                            title = patent.get("title", "")
                            body = patent.get("body", "")
                            
                            if body:
                                # 生成v1混合检索向量
                                full_text = f"{title} {body}"
                                dense_vector, sparse_vector = await g_embedding.get_embedding(full_text, use_cache=False)
                                
                                # 处理 sparse_vector 格式
                                if sparse_vector is None:
                                    sparse_vector = {}
                                elif hasattr(sparse_vector, 'indices') and hasattr(sparse_vector, 'values'):
                                    # 转换为字典格式
                                    sparse_vector = dict(zip(sparse_vector.indices, sparse_vector.values))
                                elif not isinstance(sparse_vector, dict):
                                    sparse_vector = {}
                                
                                # 生成v2扩展维度向量
                                features = await g_feature_extractor.extract_all_features(title, body)
                                
                                intermediate_buffer.append({
                                    'id': patent['id'],
                                    'dense_vector': dense_vector,
                                    'sparse_vector': sparse_vector,
                                    'vectors': {
                                        'domain_vector': features['domain_vector'],
                                        'method_vector': features['method_vector'],
                                        'application_vector': features['application_vector'],
                                        'innovation_vector': features['innovation_vector']
                                    },
                                    'extracted_texts': features['extracted_texts'],
                                    'source_id': patent['id'],
                                    'region': '',
                                    'maturity': ''
                                })
                        except Exception as e:
                            g_logger.error(f"Failed to extract features for intermediate data: {e}")
                    
                    # 当缓冲区达到最大行数时，保存到文件
                    if len(intermediate_buffer) >= MAX_ROWS_PER_FILE:
                        if not last_begin_id:
                            last_begin_id = intermediate_buffer[0]['id']
                        save_intermediate_data(intermediate_buffer, last_begin_id, resource_type)
                        last_begin_id = intermediate_buffer[-1]['id']
                        intermediate_buffer = []
                
                pbar.update(len(patents))
                pbar.set_postfix({"success": total_success, "failed": total_failed})
                
                offset += current_limit
                
                await asyncio.sleep(0.1)
        
        # 保存剩余的中间数据
        if save_intermediate and intermediate_buffer:
            if not last_begin_id:
                last_begin_id = intermediate_buffer[0]['id']
            save_intermediate_data(intermediate_buffer, last_begin_id, resource_type)
        
        if all_failed_ids:
            with open(failed_ids_file, 'w', encoding='utf-8') as f:
                for fid in all_failed_ids:
                    f.write(f"{fid}\n")
            g_logger.warning(f"Failed IDs saved to {failed_ids_file}")
        
    except Exception as e:
        g_logger.error(f"Import error: {e}")
        g_logger.error(traceback.format_exc())
    finally:
        try:
            if 'connection' in locals() and connection:
                connection.close()
        except:
            pass
    
    g_logger.info(f"Import completed:")
    g_logger.info(f"  Total: {total_count}")
    g_logger.info(f"  Success: {total_success}")
    g_logger.info(f"  Failed: {total_failed}")
    
    return total_success, total_failed

async def main():
    parser = argparse.ArgumentParser(description='Import patent data from MySQL database')
    parser.add_argument('--resource', type=str, default='patent',
                        help='Resource type (default: patent)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size for processing (default: 10)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of patents to import')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only show what would be imported, no actual inserts')
    parser.add_argument('--no-save-intermediate', action='store_true',
                        help='Do not save intermediate data files')
    parser.add_argument('--from-file', type=str, default=None,
                        help='Import from intermediate data file instead of MySQL')
    parser.add_argument('--start-id', type=str, default=None,
                        help='Start importing from patents with ID greater than this value (avoid duplicates)')
    
    args = parser.parse_args()
    
    if args.from_file:
        # 从中间数据文件导入
        await import_from_intermediate_file(
            filepath=args.from_file,
            resource_type=args.resource,
            batch_size=args.batch_size
        )
    else:
        # 从MySQL数据库导入
        await import_mysql_patent(
            resource_type=args.resource,
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run,
            save_intermediate=not args.no_save_intermediate,
            start_id=args.start_id
        )

if __name__ == "__main__":
    asyncio.run(main())
