from typing import List, Literal, Optional

from pydantic import BaseModel


class TextPart(BaseModel):
    """Описание текстовой части сообщения."""

    type: Literal["text"] = "text"
    text: str


class FilePart(BaseModel):
    """Описание файловой части сообщения."""

    type: Literal["file"] = "file"
    file: dict


Part = TextPart | FilePart


class Message(BaseModel):
    """Описание структуры сообщения."""

    role: Literal["user", "agent"]
    parts: List[Part]


class TaskRequest(BaseModel):
    """Описание запроса на выполнение задачи."""

    id: Optional[str] = None
    sessionId: Optional[str] = None
    message: Message


class TaskStatus(BaseModel):
    """Описание статуса задачи."""

    state: Literal[
        "submitted", "working", "input-required", "completed", "canceled", "failed"
    ]
    message: Optional[str] = None


class Artifact(BaseModel):
    """Описание артефакта, созданного задачей."""

    parts: List[Part]
    index: Optional[int] = 0
    append: Optional[bool] = False


class Task(BaseModel):
    """Описание структуры задачи."""

    id: str
    sessionId: Optional[str] = None
    status: TaskStatus
    messages: List[Message] = []
    artifacts: List[Artifact] = []
    history: List[dict] = []
