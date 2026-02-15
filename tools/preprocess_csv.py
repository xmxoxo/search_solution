import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import argparse
import csv
from tqdm import tqdm
from api.core.feature_extractor import g_feature_extractor
from api.utils.logger import g_logger

BATCH_SIZE = 50

async def extract_features_batch(items: list) -> list:
    """批量提取特征"""
    if not items:
        return []
    
    texts_for_extract = []
    valid_items = []
    
    for item in items:
        title = item.get("title", "")
        body = item.get("body", "")
        if body:
            texts_for_extract.append({"title": title, "body": body})
            valid_items.append(item)
    
    if not texts_for_extract:
        return [None] * len(items)
    
    try:
        features_list = await g_feature_extractor.extract_all_features_batch(texts_for_extract)
        return features_list
    except Exception as e:
        g_logger.error(f"Batch feature extraction error: {e}")
        return [None] * len(items)

async def preprocess_csv(
    input_csv: str,
    output_csv: str,
    batch_size: int = BATCH_SIZE
):
    """预处理CSV文件，提取特征文本"""
    g_logger.info(f"Preprocessing {input_csv} -> {output_csv}")
    g_logger.info(f"Batch size: {batch_size}")
    
    items = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("title", "")
            body = row.get("body", "")
            if body:
                items.append({
                    "id": row.get("id", ""),
                    "source_id": row.get("source_id", ""),
                    "title": title,
                    "body": body,
                    "region": row.get("region", ""),
                    "maturity": row.get("maturity", ""),
                    "metadata": row.get("metadata", "")
                })
    
    g_logger.info(f"Loaded {len(items)} valid items from CSV")
    
    if not items:
        g_logger.warning("No valid items to process")
        return
    
    all_extracted = []
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i+batch_size])
    
    g_logger.info(f"Processing {len(batches)} batches")
    
    with tqdm(total=len(batches), desc="Extracting features") as pbar:
        for i in range(0, len(batches)):
            batch = batches[i]
            
            try:
                features_list = await extract_features_batch(batch)
            except Exception as e:
                g_logger.error(f"Batch {i} extraction error: {e}")
                features_list = [None] * len(batch)
            
            for j, item in enumerate(batch):
                features = features_list[j]
                extracted_texts = features.get("extracted_texts", {}) if features else {}
                
                all_extracted.append({
                    "id": item["id"],
                    "source_id": item["source_id"],
                    "title": item["title"],
                    "body": item["body"],
                    "region": item["region"],
                    "maturity": item["maturity"],
                    "metadata": item["metadata"],
                    "extracted_domain": extracted_texts.get("domain", ""),
                    "extracted_method": extracted_texts.get("method", ""),
                    "extracted_application": extracted_texts.get("application", ""),
                    "extracted_innovation": extracted_texts.get("innovation", "")
                })
            
            pbar.update(1)
            pbar.set_postfix({"processed": len(all_extracted)})
    
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        fieldnames = [
            "id", "source_id", "title", "body", "region", "maturity", "metadata",
            "extracted_domain", "extracted_method", "extracted_application", "extracted_innovation"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_extracted)
    
    g_logger.info(f"Saved {len(all_extracted)} items to {output_csv}")

async def main():
    parser = argparse.ArgumentParser(description='Preprocess CSV file by extracting multi-dimension features')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for LLM processing (default: 50)')
    
    args = parser.parse_args()
    
    await preprocess_csv(args.input, args.output, args.batch_size)

if __name__ == "__main__":
    asyncio.run(main())
