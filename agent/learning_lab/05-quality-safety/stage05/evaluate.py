"""离线回归评测：意图、工具名称、订单号参数分别计分。"""

import json
from pathlib import Path


def predict(question: str) -> dict:
    normalized = question.upper()
    # 标点和自然语言中的订单号允许在正则中精确提取。
    import re
    match = re.search(r"O-\d+", normalized)
    order_id = match.group(0) if match else None
    if not order_id:
        return {"intent": "missing_order", "tool": None, "order_id": None}
    if any(word in question for word in ("售后", "退货", "退款", "申请")):
        return {"intent": "after_sale", "tool": "get_after_sale_status", "order_id": order_id}
    if any(word in question for word in ("物流", "到哪", "送到", "配送")):
        return {"intent": "logistics", "tool": "get_logistics", "order_id": order_id}
    return {"intent": "order_info", "tool": "get_order", "order_id": order_id}


def run(cases: list[dict]) -> dict:
    rows = []
    for case in cases:
        actual = predict(case["question"])
        rows.append({"question": case["question"], "expected": case,
                     "actual": actual,
                     "intent_ok": actual["intent"] == case["intent"],
                     "tool_ok": actual["tool"] == case["tool"],
                     "order_id_ok": actual["order_id"] == case["order_id"]})
    total = len(rows)
    return {"total": total,
            "intent_accuracy": sum(row["intent_ok"] for row in rows) / total if total else 1,
            "tool_accuracy": sum(row["tool_ok"] for row in rows) / total if total else 1,
            "argument_accuracy": sum(row["order_id_ok"] for row in rows) / total if total else 1,
            "passed": sum(row["intent_ok"] and row["tool_ok"] and row["order_id_ok"] for row in rows),
            "cases": rows}


if __name__ == "__main__":
    cases = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
    print(json.dumps(run(cases), ensure_ascii=False, indent=2))
