from functools import wraps

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from app.tools import (
    get_git_diff,
    list_files,
    read_file,
    run_tests,
    search_code,
)

mcp = MCPServer("CodeAgent")

def tool_with_diagnostics(tool):
    @wraps(tool)
    def wrapped(*args, **kwargs):
        try:
            return tool(*args, **kwargs)
        except ValueError as error:
            # Expected validation failures must reach the model so it can correct its arguments.
            raise ToolError(str(error)) from error
    return wrapped


for tool in (read_file, search_code, list_files, get_git_diff, run_tests):
    mcp.tool()(tool_with_diagnostics(tool))


if __name__ == "__main__":
    mcp.run()
