# 第 1 章：LLM API 与可靠服务

本章实现商品标题整理 API。模型只负责把自然语言规格整理成候选结构；服务端负责输入边界、输出 schema、业务规则、超时、并发与失败响应。先使用固定假模型学习接口，再选做真实 OpenAI-compatible 服务接入。

## 学习目标

- 掌握 system/user 消息、temperature、上下文窗口和 token 用量等 API 概念。
- 知道 Pydantic 请求校验与模型输出校验的边界不同。
- 理解 async I/O、并发上限、超时、有限重试和退避的作用。
- 能设计稳定的供应商适配接口，避免业务逻辑绑定单家 SDK。
- 能读懂 SSE 事件流；明确本章流式端点发送进度和最终字段，不冒充 token 级生成流。
- 能区分失败可重试和不可重试问题，避免无限重试放大成本。

## 概念讲解

LLM 是带随机性和延迟的远程依赖。一次请求至少有四层校验：HTTP 请求形状、模型响应 JSON 形状、字段类型/长度、业务约束。即使提示词要求“只输出 JSON”，也必须验证响应，因为模型可能截断、拒答、返回额外字段或输出错误类型。

重试只适用于可能暂时恢复的情况，例如超时、网络错误、429/5xx。认证失败、请求 schema 错误通常重试无效；业务字段不合法时也不应无限重复。重试次数必须有限，指数退避要有上限，调用必须有总超时。并发 Semaphore 控制单进程同时打到模型端的请求数；多 worker 环境还要在网关或共享存储层限流。

供应商适配用 `Normalizer` Protocol 表达业务所需能力，`FakeNormalizer` 和 `OpenAICompatibleNormalizer` 实现相同接口。这样测试不需要网络，也避免 HTTP 路由知道模型供应商细节。`ProductService` 负责供应商之外的确定性验证和失败转换。

SSE 是 HTTP 长连接上的事件帧。本章 `/normalize/stream` 会先发送 started，再发送各字段和 done；由于当前供应商适配器使用非流式完成调用，它是事件流式传输，不是模型 token 实时输出。若要 token 级流，需要把供应商 stream API 转成统一 async iterator，并设计客户端中断、错误帧和最终 JSON 校验。

## 代码阅读顺序

1. `service.py` 的 `ProductInput`、`ProductOutput`：定义业务输入/输出。
2. `Normalizer` 与两个实现：看依赖反转和本地假模型。
3. `ProductService._validate`：看严格字段集合和类型检查。
4. `ProductService.normalize`：看 Semaphore、总超时、退避、指标和失败收口。
5. `app.py`：看 Pydantic 入参、同步 JSON 接口、SSE 错误事件与 `/metrics`。

核心调用形状如下：

```python
async with self._slots:
    raw = await asyncio.wait_for(
        self.model.complete(product), timeout=self.timeout_seconds
    )
    result = self._validate(raw)
```

Semaphore 释放由 `async with` 保证，即使异常也不会永久占住并发槽。外层 API 把领域错误映射为 422（用户输入问题）或 503（模型依赖暂不可用），不把堆栈或供应商响应泄露给用户。

## 动手运行

```bash
cd agent/learning_lab
PYTHONPATH=.:01-llm-service uvicorn stage01.app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/normalize \
  -H 'Content-Type: application/json' \
  -d '{"title":"蓝牙耳机 黑色 降噪","specs":{"颜色":"黑色","连接":"蓝牙"}}'
curl -N -X POST http://127.0.0.1:8000/normalize/stream \
  -H 'Content-Type: application/json' -d '{"title":"咖啡机","specs":{}}'
curl http://127.0.0.1:8000/metrics
```

默认没有密钥也能运行。选做真实服务时，通过环境变量设置 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`；base URL 应是包含 `/chat/completions` 之前的地址。先在本地假服务或测试环境验证模型 JSON 形状，禁止把密钥写进请求样例和日志。

## 课堂实验

1. 修改 `BrokenModel` 让它第一次返回错误 schema、第二次返回合法 JSON，观察重试计数。
2. 把模型改成 `await asyncio.sleep(20)`，观察总超时和 503；再把并发上限设为 1，同时发多个请求观察排队。
3. 给响应增加 `prompt_version`、模型名、耗时与供应商返回的 usage；注意不能把原始提示词或密钥作为指标标签。
4. 比较 `temperature=0` 和非零的重复输出。即使温度为零，也不要省略服务端校验。

## 课后作业与验收

- 给 `attributes` 增加允许字段白名单，例如品牌、型号、颜色、材质；未知键进入 `missing_fields` 或拒绝。
- 为 429 与 5xx 区分重试策略，为 401/403 设计不重试的明确错误。
- 为 `/normalize/stream` 写客户端测试，断言事件顺序 started → field → done；错误时收到 error 事件。
- 加总请求耗时与重试数指标，并在 README 记录一次格式失败样例。

验收标准：无密钥离线运行；异常 JSON 不会直接返回成功；请求有界超时与有限重试；并发受控；输出仍经过确定性 schema/业务校验。
