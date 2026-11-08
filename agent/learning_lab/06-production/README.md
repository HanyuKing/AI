# 第 6 章：上线化、端到端演示与作品集

本章把前面各层组合成一个可在新环境启动的 API。整合服务提供请求关联 ID、基础限流、健康/就绪探针、身份上下文和多轮补参；Docker 镜像以非 root 用户运行。它仍是单机教学 Demo，不是可直接部署承载真实用户的生产系统。

## 学习目标

- 能从新虚拟环境安装依赖、启动 API 并跑端到端授权/越权用例。
- 解释配置、健康检查、限流、请求关联 ID 和容器安全的作用。
- 知道进程内会话/限流数据、演示身份头和 JSON 文件存储在多副本下的问题。
- 能描述模型供应商故障时的超时、有限重试、降级、队列化和成本预算方案。
- 完成一份能展示质量、成本、延迟、风险边界和故障处理的作品集说明。

## 概念讲解

可靠上线需要将客户端流量、身份系统、API 层、Agent 工作流、领域 API、知识检索、状态存储、观测平台和人工审批边界分清。Agent API 是编排入口，订单/售后领域服务仍是业务事实与写入规则的最终权威；MCP 适配层也不应绕开这些服务。

当前第 6 章使用 `X-User-Id` 简化本机练习；客户端可以伪造它，所以只能在本地信任。生产应在 API Gateway 或认证中间件验证 OAuth/OIDC token，再把校验后的 subject/tenant/roles 注入依赖上下文。不要让 Agent API 接收客户端自报身份。

限流器按用户、分钟窗口计数，帮助理解“保护下游”这一目标。它保存在 Python 进程内，多 worker、多副本各自计数，重启后清零；生产应放 API Gateway 或 Redis/Lua 等原子共享层。容量策略还包括并发上限、排队上限、请求超时、token 预算、模型路由、缓存和熔断。

Docker 使用固定基础镜像大版本、复制项目、安装运行依赖、切换非 root 用户并做 healthcheck。生产构建还应使用锁文件/哈希、扫描基础镜像、最小权限网络、只读文件系统、秘密挂载、SBOM 和镜像签名。教学 Dockerfile 为简洁没有实现这些供应链治理项。

## 代码阅读顺序

1. `stage06/app.py`：请求 schema、request ID middleware、限流、身份检查、按用户/会话隔离助手状态。
2. `stage02/assistant.py`：补充订单号时恢复上轮查询意图。
3. `stage02/tools.py` 与 `shared/domain.py`：核对最后执行工具时仍做订单归属校验。
4. `Dockerfile`：看工作目录、`PYTHONPATH`、非 root 用户和容器健康检查。
5. `tests/test_learning_lab.py`：看未认证请求拒绝、正常查询和多轮补参。

限流窗口的核心过程：

```python
while recent and recent[0] <= current - RATE_WINDOW_SECONDS:
    recent.popleft()
if len(recent) >= RATE_LIMIT:
    raise HTTPException(429, "rate limit exceeded")
recent.append(current)
```

滑动窗口能展示时效清理和拒绝边界，但读改写需要并发原子性。本实现适合单 worker 教学，不能被误认为分布式限流。

## 本地运行

```bash
cd agent/learning_lab
PYTHONPATH=06-production:02-tools-workflow uvicorn stage06.app:app --reload
```

另开终端：

```bash
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/ready
curl -X POST http://127.0.0.1:8000/assist \
  -H 'Content-Type: application/json' -H 'X-User-Id: U-100' \
  -H 'X-Request-Id: demo-001' \
  -d '{"conversation_id":"demo","message":"我的物流到哪了？"}'
curl -X POST http://127.0.0.1:8000/assist \
  -H 'Content-Type: application/json' -H 'X-User-Id: U-100' \
  -d '{"conversation_id":"demo","message":"O-1001"}'
```

第一轮会要求订单号，第二轮会使用同一会话继续查询。改成 `X-User-Id: U-200` 查询 `O-1001`，应收到拒绝。每个响应都带 `X-Request-Id` 和 `X-Process-Time-Ms`。接口文档见 `/docs`。

### Docker 演练

从仓库根目录构建，因为 Dockerfile 使用根目录作为上下文：

```bash
docker build -f agent/learning_lab/06-production/Dockerfile -t ecommerce-agent-lab .
docker run --rm -p 8000:8000 ecommerce-agent-lab
```

如果 8000 端口占用，可改成 `-p 18000:8000`。检查 `docker ps` 的 health 状态，再用上面的 curl 请求验证。

## 课堂演练：故障与容量

1. 模型适配超时：从第 1 章把超时设得很短，确认错误向用户收敛为可操作答复而不是无限等待。
2. 工具越权：查询其他用户订单，确认错误不会泄漏订单是否存在以外的敏感字段（生产需统一策略）。
3. 重复请求：讨论读取操作缓存、写请求幂等 key 和订单版本号的差别。
4. 多实例扩容：列出哪些数据现在存在进程内（会话、限流），并设计 Redis/PostgreSQL 替换接口。
5. 故障预算：设定每分钟请求量、P95 延迟、模型超时率和单会话 token 成本，说明触发降级的阈值。

## 课后作业与作品集验收

- 用已认证依赖替换 `X-User-Id`，演示错误 token、过期 token、租户不匹配和角色不足。
- 将会话 checkpoint、限流计数、审批和审计写入事务型数据库/共享缓存。
- 增加 structured logs、Prometheus/OpenTelemetry 指标与 Trace；确认能按 request ID 追踪请求。
- 做健康检查依赖超时、模型供应商故障、工具 5xx、重复消息和应用重启恢复的集成测试。
- 提交架构图、接口契约、脱敏评测结果、成本/延迟表、失败案例与复盘、已知限制和本地启动步骤。

最终验收：在干净环境按说明启动；端到端用例涵盖正常、追问、越权、依赖失败和恢复；可复现评测；没有生产密钥或真实用户数据；明确指出教学版与生产版的差距。
