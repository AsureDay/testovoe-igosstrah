# a2a/server.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Literal
import asyncio
import json
import uuid

app = FastAPI()

class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str

class FilePart(BaseModel):
    type: Literal["file"] = "file"
    file: dict  # name, mimeType, bytes или uri

Part = TextPart | FilePart

class Message(BaseModel):
    role: Literal["user", "agent"]
    parts: List[Part]

class TaskRequest(BaseModel):
    id: Optional[str] = None
    sessionId: Optional[str] = None
    message: Message

class TaskStatus(BaseModel):
    state: Literal["submitted", "working", "input-required", 
                    "completed", "canceled", "failed"]
    message: Optional[str] = None

class Artifact(BaseModel):
    parts: List[Part]
    index: Optional[int] = 0
    append: Optional[bool] = False

class Task(BaseModel):
    id: str
    sessionId: Optional[str] = None
    status: TaskStatus
    messages: List[Message] = []
    artifacts: List[Artifact] = []
    history: List[dict] = []  # внутреннее — не в спеке A2A



AGENT_CARD = {
    "name": "wiki-research-agent",
    "description": "Ищет факты в Википедии и делает логические выводы",
    "url": "https://my-agent.example.com",
    "version": "1.0.0",
    "capabilities": {
        "streaming": True,
        "pushNotifications": False,
        "stateTransitionHistory": True
    },
    "authentication": {
        "schemes": ["apiKey"]
    },
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
            "outputModes": ["text"]
        }
    ]
}

tasks: dict[str, Task] = {}

# ============ ENDPOINT'Ы ============

@app.get("/.well-known/agent.json")
def get_agent_card():
    """Другие агенты находят нас здесь"""
    return JSONResponse(content=AGENT_CARD)


@app.post("/a2a/tasks/send")
async def send_task(request: TaskRequest):
    """
    Синхронная отправка задачи.
    Агент обрабатывает и сразу возвращает результат.
    """
    task_id = request.id or str(uuid.uuid4())
    
    # Создаём задачу
    task = Task(
        id=task_id,
        sessionId=request.sessionId,
        status=TaskStatus(state="working", message="Обрабатываю..."),
        messages=[request.message]
    )
    tasks[task_id] = task
    
    try:
        result = await process_task(request.message)
        
        task.status = TaskStatus(state="completed")
        task.artifacts = [Artifact(parts=[TextPart(text=result)])]
        
    except Exception as e:
        task.status = TaskStatus(state="failed", message=str(e))
    
    return task


@app.post("/a2a/tasks/sendSubscribe")
async def send_task_stream(request: TaskRequest):
    """
    Потоковая отправка задачи.
    Возвращает SSE-ивенты с промежуточными статусами.
    """
    task_id = request.id or str(uuid.uuid4())
    # TODO: дописать эту залупу
    pass


@app.post("/a2a/tasks/get")
def get_task(request: dict):  # {"id": "task-123"}
    """Проверить статус задачи по ID"""
    task_id = request.get("id")
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


@app.post("/a2a/tasks/cancel")
def cancel_task(request: dict):  # {"id": "task-123"}
    """Отменить задачу"""
    task_id = request.get("id")
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    tasks[task_id].status = TaskStatus(state="canceled")
    return tasks[task_id]



async def process_task(message: Message) -> str:
    text = message.parts[0].text  
    
    
    
    result = await run_agent(text)
    return result


# ============ ЗАПУСК ============

if __name__ == "__main__":
    from agents.agent import ReActAgent
    from inference import InferenceModule, InferenceType
    model = InferenceModule(,InferenceType.API_AGENT_PLATFORM)
    agent = ReActAgent()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)