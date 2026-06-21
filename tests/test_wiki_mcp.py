import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_wikipedia_mcp():
    """
    Test the functionality of the Wikipedia MCP server.
    """
    import sys
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-c",
            "import sys, requests; "
            "import wikipedia_mcp.wikipedia_client; "
            "orig_get = requests.get; "
            "requests.get = lambda url, params=None, **kwargs: orig_get(url, params=params, headers={'User-Agent': 'MyIngosstrahTestWikiMCPClient/1.0 (admin@ingosstrah-test.ru)'}, **kwargs); "
            "from wikipedia_mcp.__main__ import main; "
            "sys.argv = ['wikipedia-mcp', '--language', 'ru']; "
            "main()"
        ]
    )

    print("⏳ Подключение к серверу wikipedia-mcp...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ Сессия успешно инициализирована\n")
            
            print("🔍 Доступные инструменты:")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f" - {tool.name}: {tool.description}")
            print()
                
            tool_name = "search_wikipedia"
            query = "Model Context Protocol"
            print(f"📖 Вызов '{tool_name}' с запросом '{query}'...")
            
            try:
                result = await session.call_tool(
                    tool_name,
                    arguments={"query": query}
                )
                
                print("\n✅ Ответ сервера:")
                for content in result.content:
                    if content.type == "text":
                        print(content.text)
            except Exception as e:
                print(f"❌ Произошла ошибка при вызове инструмента: {e}")

if __name__ == "__main__":
    asyncio.run(test_wikipedia_mcp())
