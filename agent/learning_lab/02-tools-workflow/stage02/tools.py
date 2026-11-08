import time
import re
from dataclasses import dataclass

from shared.domain import Forbidden, NotFound, get_after_sale_status, get_logistics, get_order


TOOL_SCHEMAS = {
    "get_order": {"order_id": "string"},
    "get_logistics": {"order_id": "string"},
    "get_after_sale_status": {"order_id": "string"},
}


class ToolError(Exception):
    pass


@dataclass(frozen=True)
class ToolResult:
    name: str
    data: dict
    elapsed_ms: int


class ReadOnlyTools:
    """工具只收 order_id；身份来自服务端上下文，绝不暴露为模型可控参数。"""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.audit: list[dict] = []

    def call(self, name: str, arguments: dict):
        if name not in TOOL_SCHEMAS:
            raise ToolError("不支持的工具")
        if set(arguments) != {"order_id"} or not isinstance(arguments["order_id"], str):
            raise ToolError("工具参数无效")
        if not re.fullmatch(r"O-\d{4,12}", arguments["order_id"]):
            raise ToolError("订单号格式无效")
        started = time.monotonic()
        try:
            if name == "get_order":
                data = get_order(arguments["order_id"], self.user_id).__dict__
            elif name == "get_logistics":
                data = get_logistics(arguments["order_id"], self.user_id)
            else:
                data = {"status": get_after_sale_status(arguments["order_id"], self.user_id)}
            elapsed_ms = round((time.monotonic() - started) * 1000)
            self.audit.append({"tool": name, "order_id": arguments["order_id"], "outcome": "ok", "elapsed_ms": elapsed_ms})
            return ToolResult(name, data, elapsed_ms)
        except Forbidden as exc:
            self.audit.append({"tool": name, "order_id": arguments["order_id"], "outcome": "forbidden"})
            raise ToolError("无权查询该订单") from exc
        except NotFound as exc:
            self.audit.append({"tool": name, "order_id": arguments["order_id"], "outcome": "not_found"})
            raise ToolError("订单不存在") from exc
