"""可选 MCP 适配示例：本地演示身份固定，不能直接用于多用户生产环境。"""

from mcp.server.fastmcp import FastMCP
from .tools import ReadOnlyTools

mcp = FastMCP("ecommerce-readonly")
_tools = ReadOnlyTools("U-100")


@mcp.tool()
def get_order(order_id: str) -> dict:
    """查询当前演示用户自己的订单。"""
    return _tools.call("get_order", {"order_id": order_id}).data


@mcp.tool()
def get_logistics(order_id: str) -> dict:
    """查询订单的实时物流演示信息。"""
    return _tools.call("get_logistics", {"order_id": order_id}).data


@mcp.tool()
def get_after_sale_status(order_id: str) -> dict:
    """查询订单售后状态。"""
    return _tools.call("get_after_sale_status", {"order_id": order_id}).data


if __name__ == "__main__":
    mcp.run()
