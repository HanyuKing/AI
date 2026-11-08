"""商品整理的模型适配与可信输出校验。"""

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass(frozen=True)
class ProductInput:
    title: str
    specs: dict[str, str]


@dataclass(frozen=True)
class ProductOutput:
    standard_title: str
    attributes: dict[str, str]
    missing_fields: list[str]


class ModelError(Exception):
    pass


class Normalizer(Protocol):
    async def complete(self, product: ProductInput) -> dict: ...


class FakeNormalizer:
    async def complete(self, product: ProductInput) -> dict:
        words = product.title.split()
        return {
            "standard_title": " ".join(words),
            "attributes": product.specs,
            "missing_fields": [key for key in ("品牌", "型号") if key not in product.specs],
        }


class OpenAICompatibleNormalizer:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 8):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def complete(self, product: ProductInput) -> dict:
        prompt = ("整理商品信息。只返回 JSON，字段为 standard_title(string), "
                  "attributes(object string->string), missing_fields(string array)。\n" +
                  json.dumps(asdict(product), ensure_ascii=False))
        payload = json.dumps({"model": self.model, "temperature": 0,
                              "messages": [{"role": "user", "content": prompt}]}).encode()

        def request() -> dict:
            req = urllib.request.Request(self.base_url + "/chat/completions", data=payload,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read())
            return json.loads(body["choices"][0]["message"]["content"])

        return await asyncio.to_thread(request)


def configured_model() -> Normalizer:
    base_url, key = os.getenv("LLM_BASE_URL"), os.getenv("LLM_API_KEY")
    if base_url and key:
        return OpenAICompatibleNormalizer(base_url, key, os.getenv("LLM_MODEL", "gpt-4o-mini"))
    return FakeNormalizer()


class ProductService:
    def __init__(self, model: Normalizer, attempts: int = 3, max_concurrency: int = 8,
                 timeout_seconds: float = 10):
        if attempts < 1 or max_concurrency < 1 or timeout_seconds <= 0:
            raise ValueError("attempts, concurrency and timeout must be positive")
        self.model, self.attempts = model, attempts
        self.timeout_seconds = timeout_seconds
        self._slots = asyncio.Semaphore(max_concurrency)
        self.metrics = {"requests": 0, "successes": 0, "failures": 0, "retries": 0}

    @staticmethod
    def _validate(raw: dict) -> ProductOutput:
        required = {"standard_title", "attributes", "missing_fields"}
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("model output schema mismatch")
        if (not isinstance(raw["standard_title"], str) or not isinstance(raw["attributes"], dict)
                or not isinstance(raw["missing_fields"], list)
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in raw["attributes"].items())
                or any(not isinstance(item, str) for item in raw["missing_fields"])):
            raise ValueError("model output field types mismatch")
        result = ProductOutput(raw["standard_title"].strip(), raw["attributes"], raw["missing_fields"])
        if not result.standard_title or len(result.standard_title) > 300:
            raise ValueError("invalid standard_title")
        if any(not key.strip() or not value.strip() for key, value in result.attributes.items()):
            raise ValueError("attributes must be non-empty strings")
        return result

    async def normalize(self, product: ProductInput) -> ProductOutput:
        result, _ = await self.normalize_with_metrics(product)
        return result

    async def normalize_with_metrics(self, product: ProductInput) -> tuple[ProductOutput, int]:
        if not product.title.strip() or len(product.title) > 300:
            raise ValueError("title must contain 1..300 characters")
        started = time.monotonic()
        error: Exception | None = None
        self.metrics["requests"] += 1
        async with self._slots:
            for attempt in range(self.attempts):
                try:
                    raw = await asyncio.wait_for(self.model.complete(product), timeout=self.timeout_seconds)
                    result = self._validate(raw)
                    self.metrics["successes"] += 1
                    return result, round((time.monotonic() - started) * 1000)
                except (asyncio.TimeoutError, urllib.error.URLError, KeyError, TypeError,
                        ValueError, json.JSONDecodeError) as exc:
                    error = exc
                    if attempt + 1 < self.attempts:
                        self.metrics["retries"] += 1
                        await asyncio.sleep(min(0.1 * (2 ** attempt), 1.0))
        self.metrics["failures"] += 1
        raise ModelError("模型暂时不可用或返回格式无效") from error

    async def stream_events(self, product: ProductInput):
        """输出进度与结构化最终结果；供应商适配器当前不承诺 token 级流式生成。"""
        yield {"event": "started", "data": {"message": "正在整理商品信息"}}
        result = await self.normalize(product)
        for field, value in result.__dict__.items():
            yield {"event": "field", "data": {"field": field, "value": value}}
        yield {"event": "done", "data": {"result": result.__dict__}}
