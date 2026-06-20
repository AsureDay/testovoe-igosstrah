from typing import Optional
from inference import InferenceModule
from tools import wikipedia_search, wikipedia_page, calculate

class ReActAgent:
    """
    Агентный модуль для решения задач по шагам с использованием парадигмы ReAct.
    """

    def __init__(self, inference_module: InferenceModule):
        self.inference_module = inference_module

    async def run(self, query: str) -> str:
        raise NotImplementedError("Метод run не реализован")

    async def self_check(self, query: str, answer: str) -> bool:
        raise NotImplementedError("Метод self_check не реализован")
