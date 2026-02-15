#!/usr/bin/env python3
#coding:utf-8

'''
author: 'xmxoxo<xmxoxo@qq.com>'
'''
import time
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm
from baselibs import api_post_data

def batch_insert(data_list):
    ''' 批量导入数据
    '''
    api_url = "http://192.168.40.64:5310/api/v1/data/insert"
    items = []
    for dat in data_list:
        sid, stitle, sbody = dat
        item = {
          "id": sid,
          "fields": {
            "title": stitle,
            "body": sbody
          },
          "raw_text_for_embedding": ""
        }
        items.append(item)

    post_data = {
      "resource_type": "tec",
      "items": items,
      "async_process": False
    }
    ret = api_post_data(api_url, post_data)
    # {'ingested_count': 1, 'failed_ids': [], 'task_id': None}
    return ret

def batch_insert_from_file(fname, batch_size=5, skip_num=0):
    '''
    '''
    # 读取CSV 
    df = pd.read_csv(fname)
    # print(df.info())
    dat_list = df.values.tolist()
    total = len(dat_list)
    # print(type(dat_list), len(dat_list))
    print(f'记录总数:{total}')
    
    success_count, finished = 0, 0
    with tqdm(total=total, desc="导入数据", ncols=80) as pbar:
        for record in dat_list[skip_num:]:
            # sid = record[0]
            # print(f'正在导入:{sid}')
            ret = batch_insert([record])
            # print(ret)
            if isinstance(ret, dict):
                ingested_count = ret.get("ingested_count", 0)
                if isinstance(ingested_count, int):
                    success_count += ingested_count

            finished += 1
            # print()
            # break;
            pbar.update(1)
            pbar.set_postfix({"success": success_count, "processed": finished})
            time.sleep(0.5)

def main(skip_num=0):
    ''' 主方法 
    '''
    fname = "data/tec_clean.csv"
    batch_insert_from_file(fname, skip_num=skip_num)


if __name__ == '__main__':
    # DEBUG INFO ERROR  CRITICAL
    logging.basicConfig(level=logging.INFO)
    import fire
    fire.Fire()

