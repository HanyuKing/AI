import os
from shared.store import JsonStore
from .workflow import ApprovalWorkflow

if __name__ == "__main__":
    flow = ApprovalWorkflow(JsonStore(os.getenv("AGENT_STATE_FILE", "/tmp/agent-learning-state.json")))
    draft = flow.create_draft("O-1002", "refund", "申请售后退款，由客服核实", "U-200")
    print("审批前:", draft)
    flow.decide(draft["draft_id"], True, "agent-demo")
    print("执行:", flow.execute(draft["draft_id"], "demo-" + draft["draft_id"]))
