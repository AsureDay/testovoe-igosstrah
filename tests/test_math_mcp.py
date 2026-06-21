import asyncio
import sys
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_math_mcp():
    """
    Тестирование работы MCP сервера calculator-mcp-server.
    """
    server_path = os.path.join(os.path.dirname(__file__), "calculator_mcp", "calculator_server.py")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_path, "--stdio"]
    )

    print("⏳ Подключение к серверу calculator-mcp-server...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ Сессия успешно инициализирована\n")
            
            print("🔍 Доступные инструменты:")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f" - {tool.name}")
            print()
                
            print("📝 Шаг 1: Решение уравнения x**2 - 4 = 0...")
            solve_res = await session.call_tool(
                "solve_equation",
                arguments={
                    "equation": "x**2 - 4 = 0"
                }
            )
            
            print("\n✅ Ответ сервера:")
            for content in solve_res.content:
                if content.type == "text":
                    print(content.text)

if __name__ == "__main__":
    asyncio.run(test_math_mcp())
