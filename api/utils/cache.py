import hashlib
import json
from typing import Any, Optional
from redis import asyncio as aioredis
from config.app_config import GBL_REDIS_CONFIG, CACHE_TTL_LLM_PARSE, CACHE_TTL_EMBEDDING, CACHE_TTL_MATCH_RESULT

class RedisCache:
    _instance: Optional["RedisCache"] = None
    
    def __new__(cls) -> "RedisCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._redis = None
        return cls._instance
    
    async def init(self):
        if self._redis is None:
            password = GBL_REDIS_CONFIG.get("password") or None
            self._redis = await aioredis.from_url(
                f"redis://{GBL_REDIS_CONFIG['host']}:{GBL_REDIS_CONFIG['port']}",
                password=password,
                db=GBL_REDIS_CONFIG["db"],
                encoding="utf-8",
                decode_responses=True
            )
    
    async def close(self):
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
    
    async def get(self, key: str) -> Optional[str]:
        if self._redis is None:
            await self.init()
        return await self._redis.get(key)
    
    async def setex(self, key: str, ttl: int, value: str):
        if self._redis is None:
            await self.init()
        await self._redis.setex(key, ttl, value)
    
    async def delete(self, key: str):
        if self._redis is None:
            await self.init()
        await self._redis.delete(key)
    
    async def get_json(self, key: str) -> Optional[Any]:
        value = await self.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def setex_json(self, key: str, ttl: Any, value: Any):
        await self.setex(key, ttl, json.dumps(value, ensure_ascii=False))
    
    @staticmethod
    def generate_key(prefix: str, content: str) -> str:
        hash_val = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"{prefix}:{hash_val}"

g_cache = RedisCache()

class CacheKeys:
    LLM_PARSE = "llm_parse"
    EMBEDDING = "embedding"
    MATCH_RESULT = "match_result"

class CacheTTL:
    LLM_PARSE = CACHE_TTL_LLM_PARSE
    EMBEDDING = CACHE_TTL_EMBEDDING
    MATCH_RESULT = CACHE_TTL_MATCH_RESULT

__all__ = ["g_cache", "CacheKeys", "CacheTTL"]
