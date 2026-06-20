import asyncio
import sys
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_math_mcp():
    """
    Тестирование работы MCP сервера sympy-mcp.
    """
    server_path = os.path.join(os.path.dirname(__file__), "sympy_mcp", "server.py")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_path]
    )

    print("⏳ Подключение к серверу sympy-mcp...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ Сессия успешно инициализирована\n")
            
            print("🔍 Доступные инструменты:")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f" - {tool.name}")
            print()
                
            print("📝 Шаг 1: Создание переменной x...")
            await session.call_tool(
                "intro",
                arguments={
                    "var_name": "x",
                    "pos_assumptions": ["real"],
                    "neg_assumptions": []
                }
            )

            print("📝 Шаг 2: Создание выражения x**2 - 4...")
            expr_res = await session.call_tool(
                "introduce_expression",
                arguments={
                    "expr_str": "x**2 - 4"
                }
            )
            expr_key = expr_res.content[0].text
            print(f"Идентификатор выражения: {expr_key}")

            print(f"📝 Шаг 3: Решение выражения {expr_key} = 0 относительно x...")
            solve_res = await session.call_tool(
                "solve_algebraically",
                arguments={
                    "expr_key": expr_key,
                    "solve_for_var_name": "x"
                }
            )
            
            print("\n✅ Ответ сервера:")
            for content in solve_res.content:
                if content.type == "text":
                    print(content.text)

if __name__ == "__main__":
    asyncio.run(test_math_mcp())
