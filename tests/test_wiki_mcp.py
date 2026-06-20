import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_wikipedia_mcp():
    server_params = StdioServerParameters(
        command="wikipedia-mcp", 
        args=[]
    )

    print("⏳ Подключение к серверу wikipedia-mcp...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Обязательный этап инициализации протокола MCP
            await session.initialize()
            print("✅ Сессия успешно инициализирована\n")
            
            # Шаг 1: Проверка списка доступных инструментов (Tools)
            print("🔍 Доступные инструменты:")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f" - {tool.name}")
            print()
                
            # Шаг 2: Тестовый вызов инструмента поиска
            tool_name = "search_wikipedia"
            query = "Model Context Protocol"
            print(f"📖 Вызов '{tool_name}' с запросом '{query}'...")
            
            try:
                # Отправляем запрос к серверу с передачей аргументов
                result = await session.call_tool(
                    tool_name, 
                    arguments={"query": query}
                )
                
                print("\n✅ Ответ сервера:")
                # Парсим ответ (выводим первые 500 символов, чтобы не засорять терминал)
                for content in result.content:
                    if content.type == "text":
                        print(content.text[:500] + "...\n\n[Текст обрезан для компактности]")
            except Exception as e:
                print(f"❌ Произошла ошибка при вызове инструмента: {e}")

if __name__ == "__main__":
    asyncio.run(test_wikipedia_mcp())