#!/usr/bin/env python3
#coding:utf-8

'''
author: 'xmxoxo<xmxoxo@qq.com>'
'''

import pymysql
from baselibs import savejson

# 数据库配置
db_config = {
    'host': 'rm-bp1r9uw2zyl43lrxpdo.mysql.rds.aliyuncs.com',
    'port': 3306,
    'user': 'readonly_1633',
    'password': '9a06_qkiLMhQq5T',
    'database': 'data_middle_group',
    'charset': 'utf8mb4'
}


def get_segment_ids_fast():
    """
    使用 OFFSET SKIP 快速获取分段点 ID
    输出格式: [[1, id1], [30000, id2], [60000, id3], ...]
    """
    conn = None
    segment_data = []  # 存储 [位置, ID] 对
    
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        
        # 先获取总记录数
        count_sql = """
        SELECT COUNT(*) FROM t_patent_trade_info 
        WHERE patent_type IS NOT NULL 
          AND patent_name IS NOT NULL 
          AND descript IS NOT NULL
        """
        cursor.execute(count_sql)
        total_count = cursor.fetchone()[0]
        print(f"总记录数: {total_count}")
        
        # 分段大小
        segment_size = 30000
        
        # 计算所有需要查询的位置点
        positions = list(range(0, total_count, segment_size))  # 0, 30000, 60000, ...
        print(f"需要查询 {len(positions)} 个分段点")
        
        # 对每个位置点，用 OFFSET 直接跳转获取 ID
        for idx, offset in enumerate(positions):
            position = offset + 1  # 人类可读的位置（从1开始）
            
            sql = """
            SELECT id FROM t_patent_trade_info 
            WHERE patent_type IS NOT NULL 
              AND patent_name IS NOT NULL 
              AND descript IS NOT NULL
            ORDER BY id
            LIMIT 1 OFFSET %s
            """
            
            cursor.execute(sql, (offset,))
            result = cursor.fetchone()
            
            if result:
                id_val = result[0]
                segment_data.append([position, id_val])
                print(f"[{idx+1}/{len(positions)}] 位置: {position}, ID: {id_val}")
            else:
                print(f"警告: OFFSET {offset} 未获取到数据")
        
        # 控制台输出最终结果（你要求的格式）
        print("\n=== 分段结果 ===")
        for pos, id_val in segment_data:
            print(f"{pos}, \"{id_val}\",")
        
        # 保存到 JSON 文件
        output_path = "../data/segment.json"
        savejson(segment_data, output_path)
        print(f"\n✅ 已保存到: {output_path}")
        print(f"数据格式: {segment_data[:3]} ... 共 {len(segment_data)} 条")
        
        return segment_data
        
    except pymysql.Error as e:
        print(f"数据库错误: {e}")
        return []
    finally:
        if conn:
            conn.close()



if __name__ == "__main__":
    get_segment_ids_fast()

