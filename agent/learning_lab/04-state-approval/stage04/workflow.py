from datetime import datetime, timedelta, timezone
import uuid

from shared.domain import Forbidden, NotFound, get_order
from shared.store import JsonStore


class ApprovalWorkflow:
    def __init__(self, store: JsonStore):
        self.store = store

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _audit(state: dict, event: str, actor: str, draft_id: str) -> None:
        state.setdefault("audit", []).append({"event": event, "actor": actor, "draft_id": draft_id,
                                               "at": datetime.now(timezone.utc).isoformat()})

    def create_draft(self, order_id: str, action: str, reason: str, requester_id: str) -> dict:
        if action not in {"refund", "reship"}:
            raise ValueError("unsupported action")
        try:
            order = get_order(order_id, requester_id)
        except (Forbidden, NotFound) as exc:
            raise ValueError(str(exc)) from exc
        if action == "refund" and order.state != "delivered":
            raise ValueError("refund draft requires a delivered order")
        if action == "reship" and order.state != "shipped":
            raise ValueError("reship draft requires a shipped order")
        state = self.store.read()
        draft_id = str(uuid.uuid4())
        now = self._now()
        draft = {"draft_id": draft_id, "order_id": order_id, "action": action,
                 "requester_id": requester_id, "reason": reason[:300],
                 "amount": order.amount if action == "refund" else 0,
                 "order_state_at_creation": order.state, "order_amount_at_creation": order.amount,
                 "created_at": now.isoformat(), "expires_at": (now + timedelta(minutes=30)).isoformat(),
                 "version": 1, "status": "pending_approval"}
        state.setdefault("drafts", {})[draft_id] = draft
        self._audit(state, "draft_created", requester_id, draft_id)
        self.store.write(state)
        return draft

    def decide(self, draft_id: str, approved: bool, operator_id: str) -> dict:
        state = self.store.read()
        draft = state.get("drafts", {}).get(draft_id)
        if not draft or draft["status"] != "pending_approval":
            raise ValueError("draft missing or already decided")
        if not operator_id or operator_id == draft["requester_id"]:
            raise ValueError("an independent reviewer is required")
        if self._now() >= datetime.fromisoformat(draft["expires_at"]):
            draft["status"] = "expired"
            draft["version"] += 1
            self._audit(state, "draft_expired", operator_id, draft_id)
            self.store.write(state)
            raise ValueError("draft expired; create a new draft")
        draft["status"] = "approved" if approved else "rejected"
        draft["operator_id"] = operator_id
        draft["version"] += 1
        self._audit(state, "draft_approved" if approved else "draft_rejected", operator_id, draft_id)
        self.store.write(state)
        return draft

    def execute(self, draft_id: str, idempotency_key: str) -> dict:
        if not idempotency_key:
            raise ValueError("idempotency key required")
        state = self.store.read()
        draft = state.get("drafts", {}).get(draft_id)
        if not draft:
            raise ValueError("draft not found")
        previous = state.setdefault("executions", {}).get(idempotency_key)
        if previous:
            if previous["draft_id"] != draft_id:
                raise ValueError("idempotency key already belongs to another draft")
            return previous
        if draft["status"] != "approved":
            raise ValueError("only an approved draft can execute")
        if self._now() >= datetime.fromisoformat(draft["expires_at"]):
            draft["status"] = "expired"
            self._audit(state, "draft_expired", "system", draft_id)
            self.store.write(state)
            raise ValueError("draft expired; create a new draft")
        try:
            order = get_order(draft["order_id"], draft["requester_id"])
        except (Forbidden, NotFound) as exc:
            raise ValueError("order is no longer accessible") from exc
        if (order.state != draft["order_state_at_creation"]
                or order.amount != draft["order_amount_at_creation"]):
            draft["status"] = "stale"
            self._audit(state, "order_changed", "system", draft_id)
            self.store.write(state)
            raise ValueError("order changed; create and review a new draft")
        result = {"draft_id": draft_id, "action": draft["action"], "status": "simulated",
                  "amount": draft["amount"]}
        state["executions"][idempotency_key] = result
        draft["status"] = "executed"
        draft["version"] += 1
        self._audit(state, "action_simulated", draft["operator_id"], draft_id)
        self.store.write(state)
        return result
