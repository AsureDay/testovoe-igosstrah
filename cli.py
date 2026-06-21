import argparse
import asyncio
import sys

from inference import InferenceModule
from agents.wiki_agent import ReActAgent
from prompt_toolkit import PromptSession

async def async_main():
    """Асинхронная основная функция для обработки аргументов командной строки и запуска агента."""
    parser = argparse.ArgumentParser(description="CLI for ReActAgent")
    parser.add_argument("query", nargs="?", type=str, help="Запрос к агенту")
    parser.add_argument("--model", type=str, default="google/gemma-4-31b-it", help="Название модели")
    parser.add_argument("--loops", type=int, default=15, help="Максимальное количество циклов агента")
    
    args = parser.parse_args()
    
    inference = InferenceModule(model_name=args.model)
    agent = ReActAgent(inference_module=inference, max_loops=args.loops)
    
    if args.query:
        response = await agent.run(args.query)
        print(response)
    else:
        session = PromptSession()
        while True:
            try:
                user_input = await session.prompt_async("> ")
                if user_input.strip().lower() in ['exit', 'quit']:
                    break
                if not user_input.strip():
                    continue
                
                response = await agent.run(user_input.strip())
                print(response)
            except (KeyboardInterrupt, EOFError):
                break

def main():
    """Точка входа в программу."""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
