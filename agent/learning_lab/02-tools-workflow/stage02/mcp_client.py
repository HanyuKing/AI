"""本地教学 Host/Client：启动 stdio server、发现工具并调用只读工具。"""

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "stage02.mcp_server"],
        env=os.environ.copy(),
    )
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            listed = await session.list_tools()
            print("发现工具:", [tool.name for tool in listed.tools])
            result = await session.call_tool("get_logistics", {"order_id": "O-1001"})
            print("工具结果:", result.model_dump(exclude_none=True))


if __name__ == "__main__":
    asyncio.run(main())
