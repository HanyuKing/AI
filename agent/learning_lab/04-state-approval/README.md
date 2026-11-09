# 第 4 章：状态管理与人工审批

本章把“Agent 建议做什么”与“业务服务真正做什么”拆开，建立可恢复、可审计、可重复提交的高风险动作流程。示例只模拟退款/补发，不连接支付或订单写 API。

## 学习目标

- 用有限状态机表达草稿的合法生命周期，而不是用自由文本触发业务动作。
- 说明审批前预览、审批人独立性、审批过期和执行前二次校验的作用。
- 知道幂等键如何避免客户端/队列重试导致重复动作，以及为何同一个 key 不能用于不同对象。
- 了解长任务 checkpoint、恢复、取消、重复投递和并发更新问题。
- 能说清本地 JSON 教学存储与数据库事务、唯一约束、乐观锁之间的差异。

## 概念讲解

退款/补发是有副作用的业务命令，不能让模型直接调用通用写工具。安全流程应先创建只读草稿，展示订单、动作、金额、原因和风险；有权限的操作人审批；执行端只接受已审批 draft ID 和幂等键；执行前重新读取订单并重算规则；最后落执行结果、领域状态和审计事件。审批期间订单可能变化，所以审批不是永久授权。

金额是领域服务计算的结果，不是模型能填写或修改的字段。示例中的全额退款从订单金额派生；动作资格根据订单状态检查。生产系统还要考虑优惠分摊、部分退款上限、退款渠道、币种精度、风控冻结和支付服务幂等。

状态机的示意：

```text
pending_approval ── approve ──> approved ── execute ──> executed
        │                           │
        ├── reject ──> rejected     ├── order changed ──> stale
        └── timeout ──> expired     └── timeout ──> expired
```

每次状态变化增加版本号和审计事件。幂等键的作用是“同一次逻辑动作的重试返回同一结果”，不是“任何操作都只做一次”。若 key 已绑定另一份草稿，必须报冲突。

## 代码阅读顺序

1. `shared/store.py`：JSON 临时文件写完后原子 replace，解释进程崩溃时避免半份 JSON；它仍没有并发事务和跨进程锁。
2. `workflow.py.create_draft`：归属/订单状态检查、金额由订单计算、记录订单快照与过期时间。
3. `decide`：只允许待审批草稿，拒绝请求者自审，超时进入 expired。
4. `execute`：先检查幂等重放，再确认已审批，重读订单状态后再记录模拟执行和审计。

读取恢复状态的入口：

```python
state = self.store.read()
draft = state.get("drafts", {}).get(draft_id)
if not draft or draft["status"] != "pending_approval":
    raise ValueError("draft missing or already decided")
```

每个状态转移都有前置状态约束，重复审批不会覆盖已有决定。演示代码将完整状态写回一个 JSON 文件，适合单进程学习，不适合并发生产写入。

## 动手运行

```bash
cd agent/learning_lab
PYTHONPATH=.:04-state-approval python -m stage04.app
```

默认演示 `U-200` 的已签收订单 `O-1002`：草稿显示领域订单金额，独立 reviewer 批准，再用同一 key 重复执行。第二次应返回相同结果。将 requester 改成 `U-100`、订单改成未送达订单，或让 requester 自己审批，观察拒绝原因。

运行状态保存在 `/tmp/agent-learning-state.json`；可以检查其 `drafts`、`executions` 和 `audit`。不要把该文件当作生产审计库。

## 课堂实验

1. 先直接执行待审批草稿，确认失败；审批后执行，再用相同 key 重试。
2. 用已被占用的 key 执行另一 draft，确认冲突不会返回错误草稿的成功结果。
3. 修改草稿中的金额或订单快照后尝试执行，思考签名/HMAC 或不可变审批 payload 的必要性。
4. 调整 `expires_at` 为过去，观察审批端将草稿置为 expired。
5. 设想两个 worker 同时执行相同 draft：指出 JSON read-modify-write 为什么会有竞态，以及数据库应如何用事务/唯一约束解决。

## 课后作业与验收

- 实现 reviewer role/权限表、审批理由、撤销和转人工；请求者与审批者不能是同一主体。
- 增加订单版本号或 `updated_at`，状态变化后审批草稿失效。
- 为状态转移写表驱动测试，覆盖所有非法边和重试路径。
- 将 JSON 存储替换成 PostgreSQL：草稿更新用事务，`idempotency_key` 加唯一约束，执行记录与 outbox/audit 同事务提交。
- 为长任务设计 checkpoint、取消标记和恢复扫描策略。

验收标准：审批前绝不能执行；用户不能伪造金额/订单归属；同一请求重复执行不重复副作用；业务状态变更会要求重新审批；关键状态有审计线索。
