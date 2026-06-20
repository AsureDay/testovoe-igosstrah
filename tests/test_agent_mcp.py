import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference import InferenceModule
from agents.wiki_agent import ReActAgent

async def test_agent_integration():
    """
    Тестирование интеграции ReActAgent с MCP серверами.
    """
    inference = InferenceModule(api_key="mock_key")
    agent = ReActAgent(inference_module=inference, max_loops=2)
    print("⏳ Тестирование запуска ReActAgent...")
    res = await agent.run("Сколько времени гепарду нужно чтобы пересечь Москву-реку?")
    print(f"Ответ агента: {res}")

if __name__ == "__main__":
    asyncio.run(test_agent_integration())
