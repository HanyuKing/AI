import re
from dataclasses import dataclass
from typing import Protocol

from .tools import ReadOnlyTools, ToolError, ToolResult


@dataclass(frozen=True)
class ToolProposal:
    name: str
    arguments: dict
    reason: str


class Planner(Protocol):
    def propose(self, message: str) -> ToolProposal: ...


class RuleBasedPlanner:
    """可替换为 LLM function-calling adapter；模型只能提出名称和参数。"""
    def propose(self, message: str) -> ToolProposal:
        if any(word in message for word in ("售后", "退货", "退款", "申请")):
            name, reason = "get_after_sale_status", "用户询问售后申请状态"
        elif any(word in message for word in ("物流", "到哪", "送到", "配送")):
            name, reason = "get_logistics", "用户询问包裹运输状态"
        else:
            name, reason = "get_order", "用户询问订单基本信息"
        match = re.search(r"O-\d+", message.upper())
        return ToolProposal(name, {"order_id": match.group(0)} if match else {}, reason)


class PlannerError(Exception):
    pass


class AfterSaleAssistant:
    def __init__(self, user_id: str, planner: Planner | None = None):
        self.tools = ReadOnlyTools(user_id)
        self.planner = planner or RuleBasedPlanner()
        self.pending_intents: dict[str, str] = {}

    def answer(self, message: str, conversation_id: str = "default") -> str:
        try:
            proposal = self.planner.propose(message)
        except PlannerError:
            return "暂时无法理解这个请求，请稍后重试或转人工。"
        order_id = proposal.arguments.get("order_id")
        if not order_id:
            self.pending_intents[conversation_id] = proposal.name
            return "请提供订单号（例如 O-1001），我再帮你查询。"
        if not re.fullmatch(r"O-\d{4,12}", order_id):
            return "订单号格式不正确，请检查后重试。"
        pending = self.pending_intents.pop(conversation_id, None)
        if pending and proposal.name == "get_order":
            proposal = ToolProposal(pending, {"order_id": order_id}, "继续上轮待补充的查询")
        try:
            result: ToolResult = self.tools.call(proposal.name, proposal.arguments)
        except ToolError as exc:
            return f"暂时无法查询：{exc}。"
        if result.name == "get_logistics":
            return f"订单 {order_id} 的物流状态：{result.data['state']}；最新信息：{result.data['last_event']}。"
        if result.name == "get_after_sale_status":
            return f"订单 {order_id} 的售后状态：{result.data['status']}。"
        return f"订单 {order_id}：{result.data['item']}，当前状态 {result.data['state']}。"
