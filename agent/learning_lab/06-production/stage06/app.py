"""整合演示 API：身份头、会话追问、请求 ID、简单限流与健康检查。"""

from collections import defaultdict, deque
import logging
import time
import uuid

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from stage02.assistant import AfterSaleAssistant

app = FastAPI(title="E-commerce After-sale Agent Demo", version="0.2.0")
logger = logging.getLogger("agent_api")
assistants: dict[tuple[str, str], AfterSaleAssistant] = {}
requests_by_user: dict[str, deque[float]] = defaultdict(deque)
RATE_LIMIT = 20
RATE_WINDOW_SECONDS = 60


class AssistRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    conversation_id: str = Field(default="default", min_length=1, max_length=80)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request failed request_id=%s path=%s", request_id, request.url.path)
        response = JSONResponse({"detail": "internal server error", "request_id": request_id}, status_code=500)
    response.headers["X-Request-Id"] = request_id
    elapsed_ms = round((time.monotonic() - started) * 1000)
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    logger.info("request complete request_id=%s method=%s path=%s status=%s elapsed_ms=%s",
                request_id, request.method, request.url.path, response.status_code, elapsed_ms)
    return response


def check_rate_limit(user_id: str, now: float | None = None) -> None:
    current = now if now is not None else time.monotonic()
    recent = requests_by_user[user_id]
    while recent and recent[0] <= current - RATE_WINDOW_SECONDS:
        recent.popleft()
    if len(recent) >= RATE_LIMIT:
        raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": "60"})
    recent.append(current)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    return {"status": "ready", "dependencies": {"orders": "demo-memory", "model": "fake-planner"}}


@app.post("/assist")
def assist(body: AssistRequest, x_user_id: str = Header(default="")):
    # 只为本地学习方便；真实部署必须由认证中间件验证 JWT/OIDC 后注入 subject。
    user_id = x_user_id.strip()
    if not user_id or len(user_id) > 100:
        raise HTTPException(401, "missing or invalid user identity")
    check_rate_limit(user_id)
    key = (user_id, body.conversation_id)
    assistant = assistants.setdefault(key, AfterSaleAssistant(user_id))
    previous_audit_count = len(assistant.tools.audit)
    answer = assistant.answer(body.message, conversation_id=body.conversation_id)
    if answer.startswith("暂时无法查询：无权"):
        raise HTTPException(403, "forbidden")
    return {"answer": answer, "conversation_id": body.conversation_id,
            "tool_audit": assistant.tools.audit[previous_audit_count:]}
