from mcp.server import MCPServer

from app.tools import (
    get_git_diff,
    list_files,
    read_file,
    run_tests,
    search_code,
)

mcp = MCPServer("CodeAgent")

mcp.tool()(read_file)
mcp.tool()(search_code)
mcp.tool()(list_files)
mcp.tool()(get_git_diff)
mcp.tool()(run_tests)


if __name__ == "__main__":
    mcp.run()
