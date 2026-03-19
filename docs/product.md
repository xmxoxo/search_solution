# 产品需求

需求描述： 写在“匹配场景.txt”文档中；


## 产品规划


构建最简洁的WEB UI 以便于调用和展示模型的结果；
提供 用于测试的成果数据集，覆盖一定的行业领域范围；
提供用于测试的测试样例，用于测试模型计算结果；
构建一个评价体系, 评测数据集，来评价相似度模型的最终得分及效果；

技术方案参考： “需求分析_qwen.md”


## 技术选型

开发语言：python;  
底层库：可优先使用：baselibs 0.1.15
接口开发使用：FastAPI; 
WEB UI使用：streamlit
数据库：
优先使用SSDB, 复杂数据要使用到关系数据库可考虑使用mysql 8; 
向量数据库使用Milvus 2.5.0 
数据缓存使用:Redis; 

模型： 
LLM模型使用:qwen3系列; 
文本嵌入模型使用：BGE-M3; 
部署了one-api进行统一接入；
统一使用openai标准库调用；

开发要求：
1. API应用层代码放在独立目录api中，默认开放端口: 5310
2. WebUI应用代码放在独立目录webui中, 默认访问端口：5320
3. 构建dockerfile容器方式用于部署，使用docker-compose文件启动整体应用；
容器中构建start.sh来启动应用或者服务；
创建start.cmd用于在windows开发环境下启动服务进行测试；
4. 各个应用将配置文件放在 config/app_config.py 中，
5. 开启logging日志模式，保存运行日志文件7天；测试时使用logging.INFO 日志级别，便于查询跟踪；
6. 容器映射配置目录：config;  日志目录：logs; 

-----------------------------------------
## 应用配置


应用配置请生成一个配置文件放到配置目录下。
相关配置信息如下：

```
# One API 配置 测试环境
ONE_API_KEY = 'sk-GQtwF5ag8p6m8wWf1232B8D5E17f4455A5C14e7a2d393aEe'
ONE_API_BASE_URL = "http://192.168.15.111:3000/v1"

# 文本嵌入模型名称
EMBEDDING_MODEL_NAME = "bge-m3:latest"

# Redis缓存 
GBL_REDIS_CONFIG = {
    "host":"192.168.15.111",
    "port": 6379,
    "password":"",
    "db":1		# 使用持久化
}

# SSDB 数据库配置
SSDB_IP = "192.168.15.111"
SSDB_PORT = 8888

# Milvus服务器地址与端口
MILVUS_HOST = '192.168.15.111'
MILVUS_PORT = 19530
```


