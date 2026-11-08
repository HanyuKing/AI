# 第 2 章：Tool Calling 与确定性工作流

本章从“聊天回答”进入 Agent 工程：系统可以提出调用订单/物流工具的建议，但工具是否存在、参数是否有效、当前用户是否有权访问，都由服务端决定。默认规划器是规则实现，提供稳定、可重复的练习基线；它与真实模型 tool-calling 的提议契约相同。

## 学习目标

- 画出一次工具调用循环：用户请求 → planner → tool proposal → 服务端校验 → 执行 → 结构化结果 → 用户答复。
- 区分模型选择工具的灵活性和领域服务执行工具的权限边界。
- 能定义单一职责、窄参数、明确错误和审计字段的工具契约。
- 理解身份上下文必须服务端注入，不能让模型选择 `user_id` 或租户。
- 为缺失信息设计多轮追问状态，并隔离不同会话。
- 解释 MCP 的 Host、Client、Server、Tools、Resources、Prompts 与普通函数调用的关系。

## 概念讲解

一个工具不是“模型可以执行任意 Python”。模型只返回工具名称和 JSON 参数，例如 `get_logistics(order_id="O-1001")`。服务端使用白名单确认工具名，用 schema 校验参数，再用已经认证的 user context 调用领域函数。服务端返回结构化结果后，模型可以负责解释；本章为了稳定性，使用普通模板构造答复。

模型提议只是一份候选计划，不能当作已授权命令。特别是 `user_id`、店铺 ID、角色、退款金额、审批结果等敏感字段不能从模型参数采信。若请求缺订单号，保存“待补齐意图”，询问订单号；用户下一轮只回答订单号时，继续原查询。会话状态需要有 TTL、最大长度和持久化策略，不能无限堆积历史对话。

工作流和 Agent 也有边界：固定步骤、校验、重试和授权条件用代码；自然语言理解或多种合理工具之间的选择可交给模型。模型不需要参与每一个 if/else。

### MCP 专项概念

MCP 是工具/资源接入协议，不是身份系统，也不替代领域 API 或工作流框架。Host 承载用户体验和 Agent；MCP Client 代表 Host 连接一个或多个 Server；Server 声明工具、资源或提示模板。REST/OpenAPI 仍适合稳定服务契约；MCP 可以把已有业务能力包装成 Agent 可发现的 tools。服务端仍要鉴权、租户过滤、限流、审计和版本兼容。

本目录提供只读 MCP Server，演示 Tools。它将演示 user 固定为 `U-100`，便于单用户 stdio 测试；这是教学限制，不是多用户身份传递方案。生产 MCP 需要把可信主体绑定到 session/连接上下文，不能增加一个可由模型随便填写的 user_id 参数。

## 代码阅读顺序

1. `assistant.py`：`Planner` 是可替换边界，`ToolProposal` 是计划，不是执行结果；`pending_intents` 保存待补齐意图。
2. `llm_planner.py`：真实 OpenAI-compatible Function Calling 请求只返回工具名和 JSON 参数，不直接执行工具。
3. `tools.py`：白名单、精确参数集合、身份注入、领域异常映射、耗时和审计。
4. `shared/domain.py`：订单归属校验发生在领域读取处，不依赖模型“遵守提示词”。
5. `mcp_server.py` 与 `mcp_client.py`：把相同只读函数包装成 MCP Tools 并由 Host/Client 发现调用，不复制业务权限规则。

需要真实模型时，在启动环境中设置 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`，再显式创建 `OpenAIToolPlanner.from_env()` 传给 `AfterSaleAssistant`。默认仍使用规则 planner，避免测试需要联网；模型调用失败会成为明确的 planner 错误，不会绕开工具权限。

工具调用故意只允许窄契约：

```python
if set(arguments) != {"order_id"}:
    raise ToolError("工具参数无效")
data = get_logistics(arguments["order_id"], self.user_id)
```

模型无法通过多带一个 `user_id` 覆盖服务端身份；订单归属失败会变成明确的 `forbidden` 审计结果。

## 动手运行

```bash
cd agent/learning_lab
PYTHONPATH=02-tools-workflow python -m stage02.app
```

试试：`我的 O-1001 到哪了？`、`O-1001 的售后怎么样？`、`物流到哪了？` 后接 `O-1001`。再把演示用户改成 `U-200`，查询 `O-1001`，确认系统拒绝越权而非编造物流状态。

### 运行 MCP Server

```bash
cd agent/learning_lab
pip install -e '.[mcp]'
PYTHONPATH=02-tools-workflow python -m stage02.mcp_server
```

这是 stdio server，终端保持运行，MCP Inspector 或兼容 Host 作为 client 连接。检查三个工具的名称、描述、参数 schema、正常返回和错误。不要把终端输出的工具结果直接视为可信指令；结果是业务数据，不是系统提示词。

仓库还提供最小 Client/Host 示例，会启动 Server 子进程、初始化会话、发现工具并调用物流工具：

```bash
cd agent/learning_lab
PYTHONPATH=02-tools-workflow python -m stage02.mcp_client
```

重点看 `ClientSession.initialize → list_tools → call_tool` 的生命周期。stdio Client 管理子进程与读写流；远程部署时传输、认证和重连方式会不同，但工具仍需在 Server 端做业务授权。

## 课堂实验

1. 增加 `search_after_sale_policy(issue_type)`，确认工具只返回政策片段和来源，不判断退款资格。
2. 构造 planner 提议 `delete_all_orders`、额外 `user_id`、错误类型 `order_id=123`，确认工具层逐个拒绝。
3. 让用户先问物流但不提供订单号，再在同一 `conversation_id` 回答订单号；使用另一 conversation 验证状态不会串线。
4. 比较直接函数、REST API、MCP Tool 的调用成本和复用方式；列出这三者分别负责的边界。

## 课后作业与验收

- 定义统一的 `ToolError` 错误码（invalid_arguments、forbidden、not_found、timeout），避免只靠中文字符串分支。
- 给会话状态增加过期时间和取消流程。
- 增加工具超时和审计 request ID；审计不要存完整隐私数据。
- 将 `AfterSaleAssistant(user_id)` 改为 `AfterSaleAssistant(user_id, OpenAIToolPlanner.from_env())`，保留同一工具执行器。把模型提出但服务端拒绝的案例加入评测集。
- 使用 Inspector 验证 MCP schema，并记录 MCP 相对普通函数调用增加了什么、没有解决什么。

验收标准：未授权订单不能查询；工具参数和工具名严格校验；模型或 planner 失败时有可解释答复；缺信息能追问并恢复；没有写工具可以越过审批。
