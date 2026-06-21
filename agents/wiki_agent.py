import re
import json
import sys
import os
import uuid
from typing import Optional, List, Dict, Any, TypedDict
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.graph import StateGraph, START, END
from inference import InferenceModule


class ReActAgent:
    """
    Агентный модуль для решения задач по шагам с использованием LangGraph и MCP сервера Wikipedia.
    """

    def __init__(self, inference_module: InferenceModule, max_loops: int = 15, system_prompt_template: Optional[str] = None):
        """
        Инициализирует агента с модулем инференса и максимальным числом циклов.
        """
        self.inference_module = inference_module
        self.max_loops = max_loops
        self.system_prompt_template = system_prompt_template


    async def run(self, query: str, return_tools: bool = False, run_id: Optional[str] = None) -> str | tuple[str, list[str]]:
        """
        Запускает цикл ReAct для ответа на запрос пользователя, подключаясь к серверам.
        """
        if not run_id:
            run_id = uuid.uuid4().hex[:8]
        try:
            return await self._run_graph(query, return_tools, run_id)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return await self._run_mock(query, str(e), return_tools, run_id)

    async def _run_graph(self, query: str, return_tools: bool = False, run_id: str = "") -> str | tuple[str, list[str]]:
        """
        Внутренний метод выполнения цикла ReAct с использованием LangGraph и MCP серверов.
        """
        wiki_params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-c",
                "import sys, requests, httpx; "
                "exec('''\n"
                "orig_get = requests.get\n"
                "requests.get = lambda url, params=None, **kwargs: orig_get(url, params=params, headers={**kwargs.pop('headers', {}), 'User-Agent': 'MyIngosstrahTestWikiMCPClient/1.0 (admin@ingosstrah-test.ru)'}, **kwargs)\n"
                "orig_send = httpx.Client.send\n"
                "httpx.Client.send = lambda self, req, **kw: (req.headers.update({'User-Agent': 'MyIngosstrahTestWikiMCPClient/1.0 (admin@ingosstrah-test.ru)'}) or orig_send(self, req, **kw))\n"
                "orig_asend = httpx.AsyncClient.send\n"
                "async def new_asend(self, req, **kw):\n"
                "    req.headers['User-Agent'] = 'MyIngosstrahTestWikiMCPClient/1.0 (admin@ingosstrah-test.ru)'\n"
                "    return await orig_asend(self, req, **kw)\n"
                "httpx.AsyncClient.send = new_asend\n"
                "from wikipedia_mcp.__main__ import main\n"
                "sys.argv = ['wikipedia-mcp', '--language', 'ru']\n"
                "main()\n"
                "''')"
            ]
        )

        calc_server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "calculator_mcp", "calculator_server.py"))
        calc_params = StdioServerParameters(
            command=sys.executable,
            args=[calc_server_path, "--stdio"]
        )

        async with stdio_client(wiki_params) as (wiki_read, wiki_write):
            async with ClientSession(wiki_read, wiki_write) as wiki_session:
                await wiki_session.initialize()

                wiki_tools = await wiki_session.list_tools()

                tools_map = {}
                for tool in wiki_tools.tools:
                    tools_map[tool.name.lower()] = (wiki_session, tool)

                async with stdio_client(calc_params) as (calc_read, calc_write):
                    async with ClientSession(calc_read, calc_write) as calc_session:
                        await calc_session.initialize()

                        calc_tools = await calc_session.list_tools()
                        for tool in calc_tools.tools:
                            tools_map[tool.name.lower()] = (calc_session, tool)

                        class AgentState(TypedDict):
                            messages: list
                            loop_count: int
                            final_answer: str
                            query: str
        
                        import logging
                        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
                        os.makedirs(log_dir, exist_ok=True)
                        
                        mcp_log_path = os.path.join(log_dir, "mcp.log")
                        mcp_logger = logging.getLogger("MCPLog")
                        if not mcp_logger.handlers:
                            fh_mcp = logging.FileHandler(mcp_log_path, encoding="utf-8")
                            fh_mcp.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
                            mcp_logger.addHandler(fh_mcp)
                            mcp_logger.setLevel(logging.INFO)
                        mcp_logger.propagate = False
        
                        tools_list_desc = []
                        for name, (session, tool) in tools_map.items():
                            tools_list_desc.append(f"- {name}: {tool.description}. Схема аргументов: {json.dumps(tool.inputSchema)}")
                        tools_text = "\n".join(tools_list_desc)
        
                        if self.system_prompt_template:
                            system_prompt = self.system_prompt_template.format(tools_text=tools_text)
                        else:
                            system_prompt = (
                                "Вы — агент по решению задач. Вы решаете задачу по шагам, используя доступные инструменты.\n"
                                "ВНИМАНИЕ: Все математические вычисления должны выполняться строго через доступные инструменты (тулзы), если это возможно.\n"
                                "На каждом шаге вы должны вывести строго в формате:\n"
                                "Thought: <ваши рассуждения о следующем шаге>\n"
                                "Action: <название_инструмента>: <аргументы в формате JSON>\n\n"
                                "Доступные инструменты:\n"
                                f"{tools_text}\n\n"
                                "Когда у вас будет окончательный ответ, выведите:\n"
                                "Thought: <ваши рассуждения>\n"
                                "Action: final_answer: <ваш окончательный ответ пользователю>\n"
                            )
        
                        async def agent_node(state: AgentState) -> dict:
                            """
                            Узел агента для генерации ответа LLM.
                            """
                            messages = [{"role": "system", "content": system_prompt}] + state["messages"]
                            response = await self.inference_module.run(messages=messages)
                            return {
                                "messages": state["messages"] + [{"role": "assistant", "content": response}],
                                "loop_count": state["loop_count"] + 1
                            }
        
                        async def tools_node(state: AgentState) -> dict:
                            """
                            Узел инструментов для вызова MCP тулзов.
                            """
                            last_message = state["messages"][-1].get("content") or ""
                            action_match = re.search(r"Action:\s*(\w+)\s*:\s*(.*?)(?=\nThought:|\n|$)", last_message, re.IGNORECASE)
                            if not action_match:
                                action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", last_message, re.DOTALL | re.IGNORECASE)
        
                            if not action_match:
                                return {"messages": state["messages"] + [{"role": "user", "content": "Ошибка: не найден формат Action: название_инструмента: аргументы."}]}
        
                            tool_name = action_match.group(1).strip().lower()
                            tool_input = action_match.group(2).strip().strip("\"'")
        
                            if tool_name not in tools_map:
                                return {"messages": state["messages"] + [{"role": "user", "content": f"Ошибка: неизвестный инструмент {tool_name}."}]}
        
                            session, tool = tools_map[tool_name]
                            arguments = {}
                            try:
                                arguments = json.loads(tool_input)
                            except Exception:
                                if tool.name in ["search_wikipedia", "wikipedia_search"]:
                                    arguments = {"query": tool_input}
                                elif tool.name in ["get_article", "get_summary"]:
                                    arguments = {"title": tool_input}
                                elif tool.name == "introduce_expression":
                                    arguments = {"expr_str": tool_input}
                                elif tool.name == "print_latex_expression":
                                    arguments = {"expr_key": tool_input}
                                elif tool.name == "simplify_expression":
                                    arguments = {"expr_key": tool_input}
                                elif tool.name == "calculate":
                                    arguments = {"expression": tool_input}
                                elif tool.name == "solve_equation":
                                    arguments = {"equation": tool_input}
                                else:
                                    return {"messages": state["messages"] + [{"role": "user", "content": f"Ошибка: не удалось распарсить аргументы инструмента {tool.name} как JSON."}]}
        
                            observation = ""
                            try:
                                mcp_logger.info(f"[{run_id}] Call: {tool.name} | Args: {arguments}")
                                result = await session.call_tool(tool.name, arguments=arguments)
                                for content in result.content:
                                    if content.type == "text":
                                        observation += content.text
                                mcp_logger.info(f"[{run_id}] Res: {observation.replace(chr(10), ' ')}")
                            except Exception as e:
                                observation = f"Ошибка при вызове инструмента: {str(e)}"
                                mcp_logger.error(f"[{run_id}] Err: {str(e).replace(chr(10), ' ')}")
        
                            return {"messages": state["messages"] + [{"role": "user", "content": f"Observation: {observation}"}]}
        
                        async def self_check_node(state: AgentState) -> dict:
                            """
                            Узел самопроверки полученного ответа.
                            """
                            last_message = state["messages"][-1].get("content") or ""
                            action_match = re.search(r"Action:\s*final_answer\s*:\s*(.*)", last_message, re.DOTALL | re.IGNORECASE)
                            if not action_match:
                                action_match = re.search(r"Action:\s*final_answer\((.*)\)", last_message, re.DOTALL | re.IGNORECASE)
                            
                            if action_match:
                                answer = action_match.group(1).strip()
                            else:
                                fa_idx = last_message.lower().find("final_answer")
                                if fa_idx != -1:
                                    answer = last_message[fa_idx + len("final_answer"):].strip(":() \n")
                                else:
                                    answer = last_message
                            is_correct = await self.self_check(state["query"], answer)
                            if is_correct:
                                return {"final_answer": answer}
                            else:
                                return {
                                    "messages": state["messages"] + [{"role": "user", "content": "Самопроверка не пройдена. Пожалуйста, проверьте рассуждения."}],
                                    "final_answer": ""
                                }
        
                        def router(state: AgentState) -> str:
                            """
                            Маршрутизатор для определения следующего шага выполнения графа.
                            """
                            if state["loop_count"] >= self.max_loops:
                                return "end"
        
                            last_message = state["messages"][-1].get("content") or ""
                            action_match = re.search(r"Action:\s*(\w+)\s*:\s*(.*?)(?=\nThought:|\n|$)", last_message, re.IGNORECASE)
                            if not action_match:
                                action_match = re.search(r"Action:\s*(\w+)\((.*?)\)", last_message, re.DOTALL | re.IGNORECASE)
        
                            if not action_match:
                                return "Validate_Answer"
        
                            tool_name = action_match.group(1).strip().lower()
                            if tool_name == "final_answer":
                                return "Validate_Answer"
        
                            return "Execute_Tool"
        
                        def self_check_router(state: AgentState) -> str:
                            """
                            Маршрутизатор после проверки ответа.
                            """
                            if state["final_answer"]:
                                return "end"
                            return "LLM_Reasoning"
        
                        workflow = StateGraph(AgentState)
                        workflow.add_node("LLM_Reasoning", agent_node)
                        workflow.add_node("Execute_Tool", tools_node)
                        workflow.add_node("Validate_Answer", self_check_node)
        
                        workflow.add_edge(START, "LLM_Reasoning")
                        workflow.add_conditional_edges(
                            "LLM_Reasoning",
                            router,
                            {
                                "Execute_Tool": "Execute_Tool",
                                "Validate_Answer": "Validate_Answer",
                                "end": END,
                                "LLM_Reasoning": "LLM_Reasoning"
                            }
                        )
                        workflow.add_edge("Execute_Tool", "LLM_Reasoning")
                        workflow.add_conditional_edges(
                            "Validate_Answer",
                            self_check_router,
                            {
                                "end": END,
                                "LLM_Reasoning": "LLM_Reasoning"
                            }
                        )
        
                        graph = workflow.compile()
        
                        try:
                            graph_png = graph.get_graph().draw_mermaid_png()
                            with open(os.path.join(log_dir, "graph.png"), "wb") as f:
                                f.write(graph_png)
                        except Exception:
                            pass
        
                        initial_state = {
                            "messages": [{"role": "user", "content": query}],
                            "loop_count": 0,
                            "final_answer": "",
                            "query": query
                        }
        
                        log_path = os.path.join(log_dir, "graph_state.log")
                        graph_logger = logging.getLogger("ReActGraph")
                        if not graph_logger.handlers:
                            fh = logging.FileHandler(log_path, encoding="utf-8")
                            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
                            graph_logger.addHandler(fh)
                            graph_logger.setLevel(logging.INFO)
                        graph_logger.propagate = False
        
                        graph_logger.info(f"[{run_id}] === НОВЫЙ ЗАПУСК ГРАФА ===")
                        graph_logger.info(f"[{run_id}] Запрос: {query}")
        
                        current_state = {}
                        try:
                            async for event in graph.astream(initial_state):
                                for node_name, node_state in event.items():
                                    graph_logger.info(f"[{run_id}] --- Узел: {node_name} ---")
                                    if "messages" in node_state and node_state["messages"]:
                                        last_msg = node_state["messages"][-1]
                                        msg_content = last_msg.get('content') or ''
                                        msg_short = msg_content.replace('\n', ' ')
                                        graph_logger.info(f"[{run_id}] Msg: [{last_msg.get('role', 'unknown')}] {msg_short}")
                                    if "final_answer" in node_state and node_state["final_answer"]:
                                        ans_short = node_state['final_answer'].replace('\n', ' ')
                                        graph_logger.info(f"[{run_id}] Final: {ans_short}")
                                    
                                    current_state.update(node_state)
                        except Exception as e:
                            graph_logger.error(f"[{run_id}] Ошибка во время выполнения графа: {str(e)}", exc_info=True)
                            raise e
        
                        if current_state and current_state.get("final_answer"):
                            ans = current_state["final_answer"]
                            if return_tools:
                                used_tools = []
                                for msg in current_state.get("messages", []):
                                    if msg.get("role") == "assistant":
                                        content = msg.get("content") or ""
                                        action_match = re.search(r"Action:\s*(\w+)", content, re.IGNORECASE)
                                        if action_match:
                                            t = action_match.group(1).strip().lower()
                                            if t != "final_answer" and t not in used_tools:
                                                used_tools.append(t)
                                return ans, used_tools
                            return ans
                        
                        err_msg = "Не удалось получить ответ в пределах лимита шагов."
                        return (err_msg, []) if return_tools else err_msg

    async def self_check(self, query: str, answer: str) -> bool:
        """
        Проверяет правильность ответа на исходный вопрос.
        """
        prompt = (
            f"Вы — эксперт-валидатор. Проверьте правильность ответа на исходный вопрос.\n"
            f"Вопрос: {query}\n"
            f"Ответ: {answer}\n\n"
            f"Выведите строго:\n"
            f"Check: TRUE (если ответ правильный)\n"
            f"или\n"
            f"Check: FALSE (если ответ содержит ошибки)\n"
            f"Reason: <причина>"
        )
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.inference_module.run(messages=messages)
            if "Check: TRUE" in response:
                return True
            return False
        except Exception:
            return False

    async def _run_mock(self, query: str, error_msg: str, return_tools: bool = False, run_id: str = "") -> str | tuple[str, list[str]]:
        """
        Возвращает сообщение об ошибке при недоступности внешнего API или сбое графа.
        """
        ans = f"[{run_id}] Ошибка API ({error_msg}). Не удалось получить ответ."
        return (ans, []) if return_tools else ans