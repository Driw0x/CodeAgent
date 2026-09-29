import pytest
from mcp.client import Client

from app.mcp_server import mcp


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_exposes_project_tools():
    async with Client(mcp) as client:
        result = await client.list_tools()

    tools = {tool.name: tool for tool in result.tools}

    assert set(tools) == {"read_file", "search_code", "list_files", "get_git_diff", "run_tests"}
    assert tools["read_file"].description
    assert tools["search_code"].description
    assert tools["list_files"].description
    assert tools["get_git_diff"].description
    assert tools["run_tests"].description
    read_schema = tools["read_file"].input_schema

    assert read_schema["type"] == "object"
    assert "path" in read_schema["properties"]
    assert "path" in read_schema["required"]


@pytest.mark.anyio
async def test_mcp_rejects_invalid_argument_type():
    async with Client(mcp) as client:
        result = await client.call_tool("list_files", {"path": ".", "max_results": "invalid"})
    assert result.is_error is True
