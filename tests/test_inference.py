import asyncio
import os
import sys

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.inference import InferenceModule, InferenceType


async def main():
    inference = InferenceModule()

    query = "Привет! Расскажи коротко, что такое MCP?"
    print(f"Запрос: {query}")
    
    try:
        response = await inference.run(query=query)
        print(f"Ответ: {response}")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(main())