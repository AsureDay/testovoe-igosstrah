import asyncio
import sys
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from app.core.inference import InferenceModule

async def main():
    wiki_params = StdioServerParameters(
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

    math_server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "calculator_mcp", "calculator_server.py"))
    math_params = StdioServerParameters(
        command=sys.executable,
        args=[math_server_path, "--stdio"]
    )

    print("Connecting to wiki...")
    async with stdio_client(wiki_params) as (wiki_read, wiki_write):
        async with ClientSession(wiki_read, wiki_write) as wiki_session:
            await wiki_session.initialize()
            print("Wiki initialized.")
            
            print("Connecting to math...")
            async with stdio_client(math_params) as (math_read, math_write):
                async with ClientSession(math_read, math_write) as math_session:
                    await math_session.initialize()
                    print("Math initialized.")
                    
                    print("Calling LLM...")
                    inf = InferenceModule()
                    res = await inf.run(query="Привет")
                    print(f"LLM Response: {res}")

if __name__ == "__main__":
    asyncio.run(main())
