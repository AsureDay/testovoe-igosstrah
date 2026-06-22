import argparse
import asyncio
import os
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prompt_toolkit import PromptSession

from app.agents.wiki_agent import ReActAgent
from app.core.inference import InferenceModule, InferenceType
from app.models.schemas import Artifact, Task, TaskRequest, TaskStatus, TextPart

tasks: dict[str, Task] = {}
server_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global server_agent
    model = InferenceModule(inference_type=InferenceType.API_AGENT_PLATFORM)
    server_agent = ReActAgent(inference_module=model)
    yield


# === FastAPI Application ===
app = FastAPI(lifespan=lifespan)

AGENT_CARD = {
    "name": "wiki-research-agent",
    "description": "Ищет факты в Википедии и делает логические выводы",
    "url": "https://my-agent.example.com",
    "version": "1.0.0",
    "capabilities": {
        "streaming": True,
        "pushNotifications": False,
        "stateTransitionHistory": True,
    },
    "authentication": {"schemes": ["apiKey"]},
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"],
    "skills": [
        {
            "id": "wiki-fact-search",
            "name": "Поиск фактов и выводы",
            "description": "Находит факты в Википедии и делает логические выводы",
            "tags": ["wikipedia", "research"],
            "examples": ["Сколько лет Москва была столицей?"],
            "inputModes": ["text"],
            "outputModes": ["text"],
        }
    ],
}


@app.get("/.well-known/agent.json")
def get_agent_card():
    """Возвращает карточку агента."""
    return JSONResponse(content=AGENT_CARD)


@app.post("/a2a/tasks/send")
async def send_task(request: TaskRequest):
    """Синхронная отправка задачи."""
    task_id = request.id or str(uuid.uuid4())

    task = Task(
        id=task_id,
        sessionId=request.sessionId,
        status=TaskStatus(state="working", message="Обрабатываю..."),
        messages=[request.message],
    )
    tasks[task_id] = task

    try:
        if not request.message.parts:
            raise ValueError("Message has no parts")
        text_parts = [p for p in request.message.parts if isinstance(p, TextPart)]
        if not text_parts:
            raise ValueError("Message has no text parts")

        text = text_parts[0].text
        result = await server_agent.run(text)

        task.status = TaskStatus(state="completed")
        task.artifacts = [Artifact(parts=[TextPart(text=result)])]
    except Exception as e:
        task.status = TaskStatus(state="failed", message=str(e))

    return task


@app.post("/a2a/tasks/sendSubscribe")
async def send_task_stream(request: TaskRequest):
    """Потоковая отправка задачи."""
    raise HTTPException(status_code=501, detail="Streaming not implemented")


@app.post("/a2a/tasks/get")
def get_task(request: dict):
    """Возвращает статус задачи по ID."""
    task_id = request.get("id")
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


@app.post("/a2a/tasks/cancel")
def cancel_task(request: dict):
    """Отменяет задачу."""
    task_id = request.get("id")
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    tasks[task_id].status = TaskStatus(state="canceled")
    return tasks[task_id]


def print_mcp_logs(mcp_logs: list) -> None:
    """Выводит логи MCP."""
    if not mcp_logs:
        return
    print("\n--- MCP Logs ---")
    for log in mcp_logs:
        print(f"Tool: {log['tool']}")
        print(f"Args: {log['args']}")
        res = str(log["result"])
        if len(res) > 500:
            print(f"Result: {res[:500]}...")
        else:
            print(f"Result: {res}")
        print("-" * 20)


async def async_cli(
    query: str = None, model_name: str = "google/gemma-4-31b-it", loops: int = 15
) -> None:
    """Асинхронная основная функция для CLI агента."""
    inference = InferenceModule(model_name=model_name)
    agent = ReActAgent(inference_module=inference, max_loops=loops)

    if query:
        ans, mcp_logs = await agent.run(query, return_mcp_logs=True)
        print_mcp_logs(mcp_logs)
        print("\n--- Final Answer ---")
        print(ans)
    else:
        session = PromptSession()
        while True:
            try:
                user_input = await session.prompt_async("> ")
                user_input = user_input.strip()
                if user_input.lower() in ["exit", "quit"]:
                    break
                if not user_input:
                    continue

                ans, mcp_logs = await agent.run(user_input, return_mcp_logs=True)
                print_mcp_logs(mcp_logs)
                print("\n--- Final Answer ---")
                print(ans)
            except (KeyboardInterrupt, EOFError):
                break


def main():
    parser = argparse.ArgumentParser(description="WikiAgent CLI & Server")
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")

    # Команда запуска сервера
    server_parser = subparsers.add_parser("server", help="Запуск FastAPI A2A сервера")
    server_parser.add_argument("--host", default="0.0.0.0", help="Хост сервера")
    server_parser.add_argument("--port", type=int, default=8000, help="Порт сервера")

    # Команда CLI
    cli_parser = subparsers.add_parser("cli", help="Запуск агента в консоли")
    cli_parser.add_argument("query", nargs="?", type=str, help="Запрос к агенту")
    cli_parser.add_argument(
        "--model", type=str, default="google/gemma-4-31b-it", help="Название модели"
    )
    cli_parser.add_argument(
        "--loops", type=int, default=15, help="Максимальное количество циклов агента"
    )

    args = parser.parse_args()

    if args.command == "server":
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
        if os.path.exists(log_dir):
            for filename in os.listdir(log_dir):
                if filename.endswith(".log"):
                    try:
                        os.remove(os.path.join(log_dir, filename))
                    except OSError:
                        pass
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "cli":
        asyncio.run(
            async_cli(query=args.query, model_name=args.model, loops=args.loops)
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
