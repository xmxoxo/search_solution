import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import argparse
from tqdm import tqdm
from pymilvus import connections, Collection, utility
from api.core.embedding import g_embedding
from api.core.milvus_client import g_milvus
from api.core.feature_extractor import g_feature_extractor
from api.utils.logger import g_logger
from config.app_config import RESOURCE_TYPES, MILVUS_HOST, MILVUS_PORT

BATCH_SIZE = 50
MAX_RETRIES = 5
RETRY_DELAY = 2
REQUEST_DELAY = 0.1
MAX_PARALLEL_BATCHES = 3

def connect_milvus():
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
    except Exception as e:
        pass

def get_v1_data(resource_type: str, batch_size: int = 1000) -> list:
    connect_milvus()
    
    collection_name = f"ime_{resource_type}"
    if not utility.has_collection(collection_name):
        g_logger.warning(f"V1 collection {collection_name} not exists")
        return []
    
    collection = Collection(collection_name)
    collection.load()
    
    all_data = []
    last_id = ""
    
    while True:
        expr = f"id > '{last_id}'" if last_id else "id != ''"
        
        results = collection.query(
            expr=expr,
            output_fields=["id", "source_id", "region", "maturity", "metadata"],
            limit=batch_size
        )
        
        if not results:
            break
        
        for item in results:
            all_data.append(item)
            last_id = item["id"]
        
        if len(results) < batch_size:
            break
    
    g_logger.info(f"Found {len(all_data)} records in v1 collection {collection_name}")
    return all_data

async def process_batch_optimized(batch_data: list, resource_type: str, existing_ids: set = None) -> tuple:
    """批量处理：使用批量LLM和批量向量化"""
    if not batch_data:
        return 0, 0, []
    
    if existing_ids is None:
        existing_ids = set()
    
    items_for_extract = []
    valid_items = []
    skipped_count = 0
    
    for item in batch_data:
        item_id = item.get('id', '')
        if item_id in existing_ids:
            skipped_count += 1
            continue
        
        title = item.get("metadata", {}).get("title", "")
        body = item.get("metadata", {}).get("body", "")
        if body:
            items_for_extract.append({"title": title, "body": body})
            valid_items.append(item)
    
    if not items_for_extract:
        return 0, 0, []
    
    try:
        features_list = await g_feature_extractor.extract_all_features_batch(items_for_extract)
    except Exception as e:
        g_logger.error(f"Batch feature extraction error: {e}")
        return 0, 0, []
    
    insert_items = []
    for i, item in enumerate(valid_items):
        features = features_list[i]
        if features is None:
            continue
        
        insert_items.append({
            'id': item['id'],
            'vectors': {
                'domain_vector': features['domain_vector'],
                'method_vector': features['method_vector'],
                'application_vector': features['application_vector'],
                'innovation_vector': features['innovation_vector']
            },
            'extracted_texts': features['extracted_texts'],
            'source_id': item.get('source_id', item['id']),
            'region': item.get('region', ''),
            'maturity': item.get('maturity', ''),
            'metadata': item.get('metadata', {})
        })
    
    if not insert_items:
        return 0, 0, []
    
    for attempt in range(MAX_RETRIES):
        try:
            inserted_count = await g_milvus.insert_batch_v2(
                resource_type=resource_type,
                items=insert_items,
                skip_duplicate_check=True
            )
            
            failed_count = len(valid_items) - inserted_count
            failed_ids = []
            if failed_count > 0:
                all_ids = set(item['id'] for item in valid_items)
                success_ids = set(item['id'] for item in insert_items)
                failed_ids = list(all_ids - success_ids)
            
            return inserted_count, failed_count, failed_ids
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (attempt + 1)
                g_logger.warning(f"Batch insert retry {attempt+1}/{MAX_RETRIES} after {delay}s: {e}")
                await asyncio.sleep(delay)
            else:
                g_logger.error(f"Batch v2 insert error after retries: {e}")
                return 0, 0, []

async def process_batches_parallel(batches: list, resource_type: str) -> tuple:
    """并行处理多个批次"""
    total_success = 0
    total_failed = 0
    all_failed_ids = []
    
    for i in range(0, len(batches), MAX_PARALLEL_BATCHES):
        parallel_batches = batches[i:i+MAX_PARALLEL_BATCHES]
        
        tasks = [
            process_batch_optimized(batch, resource_type)
            for batch in parallel_batches
        ]
        
        results = await asyncio.gather(*tasks)
        
        for success, failed, ids in results:
            total_success += success
            total_failed += failed
            all_failed_ids.extend(ids)
        
        await asyncio.sleep(REQUEST_DELAY)
    
    return total_success, total_failed, all_failed_ids

async def migrate_v1_to_v2(resource_type: str, dry_run: bool = False, start_index: int = 0):
    g_logger.info(f"Starting migration from v1 to v2 for resource type: {resource_type}")
    g_logger.info(f"Batch size: {BATCH_SIZE}, Parallel batches: {MAX_PARALLEL_BATCHES}, Start index: {start_index}")
    
    v1_data = get_v1_data(resource_type)
    if not v1_data:
        g_logger.info(f"No data to migrate for {resource_type}")
        return
    
    if start_index > 0:
        v1_data = v1_data[start_index:]
        g_logger.info(f"Resuming from index {start_index}, remaining: {len(v1_data)}")
    
    total_count = len(v1_data)
    success_count = 0
    failed_count = 0
    skipped_count = 0
    failed_ids = []
    
    connect_milvus()
    from pymilvus import utility
    
    existing_ids_v2a = set()
    existing_ids_v2b = set()
    
    collection_name_v2a = f"ime_{resource_type}_v2a"
    collection_name_v2b = f"ime_{resource_type}_v2b"
    
    if utility.has_collection(collection_name_v2a):
        try:
            coll_a = Collection(collection_name_v2a)
            coll_a.load()
            last_id = ""
            while True:
                expr = f"id > '{last_id}'" if last_id else "id != ''"
                results = coll_a.query(
                    expr=expr,
                    output_fields=["id"],
                    limit=1000
                )
                if not results:
                    break
                for r in results:
                    existing_ids_v2a.add(r["id"])
                    last_id = r["id"]
                if len(results) < 1000:
                    break
            g_logger.info(f"Found {len(existing_ids_v2a)} existing IDs in {collection_name_v2a}")
        except Exception as e:
            g_logger.warning(f"Failed to get existing IDs from {collection_name_v2a}: {e}")
    
    if utility.has_collection(collection_name_v2b):
        try:
            coll_b = Collection(collection_name_v2b)
            coll_b.load()
            last_id = ""
            while True:
                expr = f"id > '{last_id}'" if last_id else "id != ''"
                results = coll_b.query(
                    expr=expr,
                    output_fields=["id"],
                    limit=1000
                )
                if not results:
                    break
                for r in results:
                    existing_ids_v2b.add(r["id"])
                    last_id = r["id"]
                if len(results) < 1000:
                    break
            g_logger.info(f"Found {len(existing_ids_v2b)} existing IDs in {collection_name_v2b}")
        except Exception as e:
            g_logger.warning(f"Failed to get existing IDs from {collection_name_v2b}: {e}")
    
    all_existing_ids = existing_ids_v2a | existing_ids_v2b
    g_logger.info(f"Total existing IDs in v2 collections: {len(all_existing_ids)}")
    
    batches = []
    for i in range(0, total_count, BATCH_SIZE):
        batches.append(v1_data[i:i+BATCH_SIZE])
    
    g_logger.info(f"Prepared {len(batches)} batches for processing")
    
    if dry_run:
        g_logger.info(f"Dry run: would process {len(batches)} batches, {total_count} records")
        return
    
    with tqdm(total=len(batches), desc=f"Migrating {resource_type}") as pbar:
        for i in range(0, len(batches), MAX_PARALLEL_BATCHES):
            parallel_batches = batches[i:i+MAX_PARALLEL_BATCHES]
            
            tasks = [
                process_batch_optimized(batch, resource_type, all_existing_ids)
                for batch in parallel_batches
            ]
            
            results = await asyncio.gather(*tasks)
            
            for success, failed, ids in results:
                success_count += success
                failed_count += failed
                failed_ids.extend(ids)
            
            pbar.update(len(parallel_batches))
            pbar.set_postfix({"success": success_count, "failed": failed_count})
            
            await asyncio.sleep(REQUEST_DELAY)
            
            if (i + MAX_PARALLEL_BATCHES) % 30 == 0:
                g_logger.info(f"Progress: {i+MAX_PARALLEL_BATCHES}/{len(batches)} batches, success: {success_count}, failed: {failed_count}")
    
    g_logger.info(f"Migration completed for {resource_type}:")
    g_logger.info(f"  Total: {total_count}")
    g_logger.info(f"  Skipped (existing): {len(all_existing_ids)}")
    g_logger.info(f"  Success: {success_count}")
    g_logger.info(f"  Failed: {failed_count}")
    
    if failed_ids:
        g_logger.warning(f"Failed IDs: {failed_ids[:20]}")
        if len(failed_ids) > 20:
            g_logger.warning(f"  ... and {len(failed_ids) - 20} more")
    
    return success_count, failed_count

async def main():
    parser = argparse.ArgumentParser(description='Migrate data from v1 collection to v2 collections')
    parser.add_argument('--resource', type=str, required=True, choices=RESOURCE_TYPES,
                        help='Resource type to migrate')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only show what would be migrated, no actual inserts')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Start from specific index (for resuming interrupted migration)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for processing (default: 50)')
    parser.add_argument('--parallel', type=int, default=3,
                        help='Number of batches to process in parallel (default: 3)')
    
    args = parser.parse_args()
    
    global BATCH_SIZE, MAX_PARALLEL_BATCHES
    BATCH_SIZE = args.batch_size
    MAX_PARALLEL_BATCHES = args.parallel
    
    await migrate_v1_to_v2(args.resource, args.dry_run, args.start_index)

if __name__ == "__main__":
    asyncio.run(main())
