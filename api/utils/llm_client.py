import json
from typing import Any, Dict, Optional
from openai import AsyncOpenAI
from config.app_config import ONE_API_KEY, ONE_API_BASE_URL, LLM_MODEL_NAME
from api.utils.logger import g_logger

class LLMClient:
    _instance: Optional["LLMClient"] = None
    
    def __new__(cls) -> "LLMClient":
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
    
    async def chat_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> str:
        if self._client is None:
            self.init()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self._client.chat.completions.create(
                model=model or LLM_MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            g_logger.error(f"LLM chat completion error: {e}")
            raise
    
    async def parse_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        result = await self.chat_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        try:
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            return json.loads(result)
        except json.JSONDecodeError as e:
            g_logger.error(f"Failed to parse LLM response as JSON: {result[:200]}")
            raise

g_llm = LLMClient()

__all__ = ["g_llm", "LLMClient"]
