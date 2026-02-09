from typing import Dict, List, Optional, Tuple
from openai import AsyncOpenAI
from config.app_config import ONE_API_KEY, ONE_API_BASE_URL, EMBEDDING_MODEL_NAME
from api.utils.logger import g_logger
from api.utils.cache import g_cache, CacheKeys, CacheTTL

class EmbeddingClient:
    _instance: Optional["EmbeddingClient"] = None
    
    def __new__(cls) -> "EmbeddingClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._client = None
        return cls._instance
    
    def init(self):
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=ONE_API_KEY,
                base_url=ONE_API_BASE_URL
            )
    
    async def get_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[List[float], Optional[Dict[int, float]]]:
        if use_cache:
            cache_key = g_cache.generate_key(CacheKeys.EMBEDDING, text)
            cached = await g_cache.get_json(cache_key)
            if cached:
                return cached.get("dense"), cached.get("sparse")
        
        if self._client is None:
            self.init()
        
        try:
            response = await self._client.embeddings.create(
                input=text,
                model=model or EMBEDDING_MODEL_NAME,
                encoding_format="float"
            )
            
            dense_vector = response.data[0].embedding
            sparse_vector = None
            
            if hasattr(response.data[0], 'sparse_embedding'):
                sparse_vector = response.data[0].sparse_embedding
            elif hasattr(response.data[0], 'sparse'):
                sparse_vector = response.data[0].sparse
            
            if use_cache:
                cache_key = g_cache.generate_key(CacheKeys.EMBEDDING, text)
                await g_cache.setex_json(
                    cache_key,
                    CacheTTL.EMBEDDING,
                    {"dense": dense_vector, "sparse": sparse_vector}
                )
            
            return dense_vector, sparse_vector
        except Exception as e:
            g_logger.error(f"Embedding error: {e}")
            raise
    
    async def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Tuple[List[float], Optional[Dict[int, float]]]]:
        results = []
        for text in texts:
            embedding = await self.get_embedding(text, model, use_cache)
            results.append(embedding)
        return results

g_embedding = EmbeddingClient()

__all__ = ["g_embedding", "EmbeddingClient"]
