import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from shared.settings import Settings

app = FastAPI(title="E-commerce Agent Learning Lab", version="0.1.0")
settings = Settings.from_env()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("learning_lab")


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "stage": "00-environment"}


@app.get("/ready")
def ready() -> dict[str, str]:
    # 此骨架没有外部依赖；后续应在这里检查数据库、队列等依赖是否可用。
    return {"status": "ready", "environment": settings.app_env}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
