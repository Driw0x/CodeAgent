import pytest
from mcp.client import Client

from app.mcp_server import mcp


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


@pytest.mark.anyio
@pytest.mark.parametrize("name,arguments,diagnostic", [
    ("list_files", {"path": "../"}, "Path must stay inside the project."),
    ("search_code", {"query": ""}, "query cannot be empty."),
    ("run_tests", {"timeout_seconds": 0}, "timeout_seconds must be between 1 and 300."),
])
async def test_mcp_preserves_expected_validation_diagnostics(name, arguments, diagnostic):
    async with Client(mcp) as client:
        result = await client.call_tool(name, arguments)
    assert result.is_error is True
    assert diagnostic in "\n".join(block.text for block in result.content)
