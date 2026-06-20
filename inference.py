import os
import openai
from enum import Enum
from typing import List, Dict, Any, Optional

class InferenceType(Enum):
    API_AGENT_PLATFORM = "api_agent_platform"
    LOCAL = "local"


class InferenceModule:
    def __init__(
        self,
        model_name: str = "qwen/qwen3.7-plus",
        api_key: str = "",
        inference_type: InferenceType = InferenceType.API_AGENT_PLATFORM
    ):
        self.model_name = model_name
        self.inference_type = inference_type

        # Если передан ключ в виде переменной окружения (например, $AGENTPLATFORM_KEY), извлекаем его значение
        if api_key.startswith("$"):
            api_key = os.environ.get(api_key[1:], "")
        
        # Если API-ключ не передан, пытаемся получить его из AGENTPLATFORM_KEY
        if not api_key:
            api_key = os.environ.get("AGENTPLATFORM_KEY", "")

        # Установка базового URL в зависимости от типа инференса
        if self.inference_type == InferenceType.API_AGENT_PLATFORM or (isinstance(self.inference_type, str) and self.inference_type == "api_agent_platform"):
            base_url = "https://api.agentplatform.ru/v1"
        else:
            base_url = os.environ.get("LOCAL_INFERENCE_URL", "http://localhost:8000/v1")
            if not api_key:
                api_key = "local"

        # Инициализация асинхронного клиента OpenAI
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )

    async def run(self, query: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None) -> str:
        # Проверка и формирование списка сообщений для отправки в модель
        if messages is not None:
            formatted_messages = messages
        elif query is not None:
            formatted_messages = [{"role": "user", "content": query}]
        else:
            raise ValueError("Необходимо передать либо query, либо messages")

        # Запрос к API
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages
        )
        
        # Возвращаем текстовый ответ модели
        return response.choices[0].message.content 