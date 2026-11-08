"""虚构电商售后领域数据和确定性服务。"""

from dataclasses import dataclass


class NotFound(Exception):
    pass


class Forbidden(Exception):
    pass


@dataclass(frozen=True)
class Order:
    order_id: str
    user_id: str
    item: str
    state: str
    amount: int
    tracking_no: str


ORDERS = {
    "O-1001": Order("O-1001", "U-100", "无线耳机", "shipped", 29900, "SF1001"),
    "O-1002": Order("O-1002", "U-200", "咖啡机", "delivered", 89900, "SF1002"),
}
LOGISTICS = {
    "O-1001": {"state": "运输中", "last_event": "包裹已到达杭州转运中心"},
    "O-1002": {"state": "已签收", "last_event": "昨日 14:20 已由本人签收"},
}
AFTER_SALE = {"O-1001": "无售后申请", "O-1002": "退货申请审核中"}


def get_order(order_id: str, user_id: str) -> Order:
    order = ORDERS.get(order_id)
    if order is None:
        raise NotFound("订单不存在")
    if order.user_id != user_id:
        raise Forbidden("无权访问该订单")
    return order


def get_logistics(order_id: str, user_id: str) -> dict[str, str]:
    get_order(order_id, user_id)
    return LOGISTICS[order_id]


def get_after_sale_status(order_id: str, user_id: str) -> str:
    get_order(order_id, user_id)
    return AFTER_SALE[order_id]


POLICIES = [
    {"id": "policy-return-v1", "tenant": "shop-a", "version": 1, "effective": "2026-01-01", "topic": "退货", "text": "商品签收后七日内，商品完好且符合平台规则时可申请退货；最终资格以售后服务校验为准。"},
    {"id": "policy-shipping-v1", "tenant": "shop-a", "version": 1, "effective": "2026-01-01", "topic": "物流", "text": "物流超过承诺时效仍未送达时，可联系商家核查；物流状态以承运商实时信息为准。"},
    {"id": "policy-return-secret", "tenant": "shop-b", "version": 1, "effective": "2026-01-01", "topic": "退货", "text": "shop-b 的内部规则，不可被 shop-a 用户检索。"},
]
