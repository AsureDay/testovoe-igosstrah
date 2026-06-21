import os
import json
from datetime import datetime
import openai
from enum import Enum
from typing import List, Dict, Any, Optional

class InferenceType(Enum):
    API_AGENT_PLATFORM = "api_agent_platform"
    LOCAL = "local"


class InferenceModule:
    def __init__(
        self,
        model_name: str = "google/gemma-4-31b-it",
        api_key: str = "",
        inference_type: InferenceType = InferenceType.API_AGENT_PLATFORM
    ):
        self.model_name = model_name
        if isinstance(inference_type, str):
            inference_type = InferenceType(inference_type)
        self.inference_type = inference_type

        if not api_key:
            api_key = os.environ.get("AGENTPLATFORM_KEY", "")

        if not api_key:
            for env_path in [os.path.expanduser("~/.env"), ".env"]:
                if os.path.exists(env_path):
                    try:
                        with open(env_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip().startswith("AGENTPLATFORM_KEY="):
                                    api_key = line.split("=", 1)[1].strip().strip("'\"")
                                    break
                    except Exception:
                        pass
                if api_key:
                    break

        if api_key and api_key.startswith("$"):
            api_key = os.environ.get(api_key[1:], "")
        
        if self.inference_type == InferenceType.API_AGENT_PLATFORM:
            base_url = "https://api.agentplatform.ru/v1"
        else:
            base_url = os.environ.get("LOCAL_INFERENCE_URL", "http://localhost:8000/v1")
            if not api_key:
                api_key = "local"

        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )

    async def run(self, query: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None, response_format: Optional[Dict[str, Any]] = None) -> str:
        if messages is not None:
            formatted_messages = messages
        elif query is not None:
            formatted_messages = [{"role": "user", "content": query}]
        else:
            raise ValueError("Необходимо передать либо query, либо messages")
        
        kwargs = {
            "model": self.model_name,
            "messages": formatted_messages,
            "timeout": 300.0
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        try:
            response = await self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            self._log_request(formatted_messages, content)
            return content
        except Exception as e:
            self._log_request(formatted_messages, None, str(e))
            raise e

    def _log_request(
        self,
        messages: List[Dict[str, str]],
        response: Optional[str],
        error: Optional[str] = None
    ) -> None:
        """
        Записывает информацию о запросе в файл.
        """
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "llm_requests.log")
        
        if messages:
            content = messages[-1].get("content", "")
            msg_summary = content
        else:
            msg_summary = ""
        resp_summary = response
        
        log_entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model_name,
            "req": msg_summary,
            "res": resp_summary,
            "err": error
        }
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass 