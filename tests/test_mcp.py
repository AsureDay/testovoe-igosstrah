import asyncio
import importlib
import os
import pathlib
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_all_mcp_get_tools():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_dir = pathlib.Path(project_root) / "app" / "tools"
    server_files = list(base_dir.rglob("*_server.py"))

    assert len(server_files) > 0, "No MCP servers found"

    for server_file in server_files:
        rel_path = server_file.relative_to(project_root)
        module_path = str(rel_path).replace(os.sep, ".").replace(".py", "")

        module = importlib.import_module(module_path)

        assert hasattr(module, "app"), f"Module {module_path} missing 'app' object"
        app = module.app

        tools = asyncio.run(app.list_tools())

        assert tools is not None, f"list_tools() returned None for {module_path}"
        assert len(tools) > 0, f"list_tools() returned empty list for {module_path}"
