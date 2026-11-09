# 第 0 章：工程环境与服务骨架

本章的任务是把一个 AI 应用做成普通、可检查的后端服务。模型还没有出场：在引入不确定的外部依赖前，先建立配置、日志、错误响应、健康检查和可重复启动方式。

## 学习目标

- 能创建隔离的 Python 虚拟环境并以项目方式管理依赖。
- 区分配置、代码常量和秘密；知道为什么不把密钥放进仓库。
- 区分存活检查 `/health` 与就绪检查 `/ready`。
- 为每个 HTTP 请求生成/传播 request ID，并记录耗时而不记录敏感正文。
- 能解释 FastAPI 的路由、Pydantic 请求校验、中间件和 Uvicorn 进程各自负责什么。

## 概念讲解

AI 服务和普通 API 服务的运行底座相同。`pyproject.toml` 声明依赖和 Python 版本；虚拟环境隔离项目依赖；应用启动时从环境读取配置；反向代理或容器探针调用健康端点。配置应尽早校验，例如端口超出范围应在进程启动时失败，而不是等到部署后才报错。

`/health` 只说明进程能够处理 HTTP 请求，常用于容器存活探针。`/ready` 表示服务已准备好接流量，通常还要检查数据库、队列、模型网关等关键依赖。两个端点不能混为一谈：依赖短暂不可用时可能保持进程存活，但暂时不应接新流量。

Request ID 用于把网关日志、应用日志和一次工具调用串起来。它是关联标识，不是认证凭据。请求 ID 不存在时服务端生成；响应中返回它，日志记录它和耗时。日志应记录路径、状态、耗时等元数据，避免把完整用户消息、地址、令牌写进去。

## 代码导读

从 `shared/settings.py` 看集中配置与边界校验：

```python
settings = Settings.from_env()
if not 1 <= settings.port <= 65535:
    raise ValueError("APP_PORT must be between 1 and 65535")
```

再看 `stage00/app.py` 的 HTTP 中间件。它在响应上设置 `X-Request-Id` 和 `X-Process-Time-Ms`。业务路由不需要重复写这些横切逻辑。当前 `/ready` 返回“无外部依赖”的演示状态；加数据库后，应在这里做有超时的轻量探活。

## 动手运行

以下命令分两种工作目录，选择一种执行，不要混用相对路径。

### 方式 A：从课程根目录运行（推荐）

```bash
cd agent/learning_lab
pip install -e '.[dev]'
PYTHONPATH=.:00-environment APP_PORT=8010 python -m stage00.app
```

### 方式 B：当前目录已经是 `00-environment`

```bash
pip install -e ..
PYTHONPATH=..:. APP_PORT=8010 python -m stage00.app
```

这里 `..` 是课程根目录 `learning_lab`（包含 `pyproject.toml` 和 `shared/`），`.` 是当前阶段目录（包含 `stage00/`）。因此不要在 `00-environment` 目录执行 `pip install -e .`，也不要在那里使用根目录版本的 `PYTHONPATH=.:00-environment`。

开发时从课程根目录启动 Uvicorn：`PYTHONPATH=.:00-environment uvicorn stage00.app:app --reload --port 8010`；若当前目录在 `00-environment`，则使用 `PYTHONPATH=..:. uvicorn stage00.app:app --reload --port 8010`。

另开终端检查：

```bash
curl -i http://127.0.0.1:8010/health
curl -i -H 'X-Request-Id: lesson-001' http://127.0.0.1:8010/ready
```

第二个响应应带回 `X-Request-Id: lesson-001` 和处理耗时头。访问 `http://127.0.0.1:8010/docs` 查看自动生成的 OpenAPI 文档。

## 课堂实验

1. 把 `APP_PORT=99999` 传入进程，观察配置是否在应用启动前失败。
2. 去掉请求 ID，再请求一次，确认服务会生成 UUID 并回传。
3. 在 `/ready` 中模拟一个依赖不可用状态，思考探针应返回 503 还是 200，以及部署平台会如何处理。
4. 让中间件只记录 `request_id`、路径、状态和耗时。确认日志中没有请求正文。

## 课后作业与验收

- 新增一个 `/version` 端点，返回应用版本和环境名，但不能返回密钥或完整环境变量。
- 将配置测试扩展到错误端口、负数超时和零并发上限。
- 为一个请求添加自定义 request ID，断言请求头和响应头一致。
- 用自己的话说明 liveness/readiness 的差别，并画出浏览器 → Uvicorn → FastAPI middleware → route 的调用顺序。

验收标准：新建虚拟环境后能启动服务；无效配置会快速失败；请求关联 ID 可贯穿响应；探活接口不会泄露部署秘密。
