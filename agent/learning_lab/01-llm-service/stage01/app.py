import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .service import ModelError, ProductInput, ProductService, configured_model

app = FastAPI(title="商品信息整理 API")
service = ProductService(configured_model())


class NormalizeRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    specs: dict[str, str] = Field(default_factory=dict)


@app.post("/normalize")
async def normalize(body: NormalizeRequest):
    try:
        result, latency_ms = await service.normalize_with_metrics(ProductInput(body.title, body.specs))
        return {"result": result.__dict__, "model": getattr(service.model, "model", "fake"),
                "latency_ms": latency_ms}
    except (ModelError, ValueError) as exc:
        raise HTTPException(503 if isinstance(exc, ModelError) else 422, str(exc)) from exc


@app.post("/normalize/stream")
async def normalize_stream(body: NormalizeRequest):
    async def events():
        try:
            async for event in service.stream_events(ProductInput(body.title, body.specs)):
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
        except (ModelError, ValueError) as exc:
            yield f"event: error\ndata: {json.dumps({'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/metrics")
def metrics():
    """教学用进程内计数器；多 worker 部署时应使用共享指标后端。"""
    return dict(service.metrics)


if __name__ == "__main__":
    import asyncio
    print(asyncio.run(service.normalize(ProductInput("蓝牙耳机 黑色 降噪", {"颜色": "黑色"}))))
