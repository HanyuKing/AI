"""最小结构化 trace；只记录允许的元数据，不保存原始对话。"""

import re
import time
import uuid


REDACTIONS = (
    (re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[PHONE]"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]+=*"), "Bearer [TOKEN]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "[API_KEY]"),
)


def redact(text: str) -> str:
    for pattern, replacement in REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


class Trace:
    def __init__(self, request_id: str | None = None, model: str = "fake", prompt_version: str = "v1"):
        self.trace_id = request_id or str(uuid.uuid4())
        self.model = model
        self.prompt_version = prompt_version
        self.started = time.monotonic()
        self.steps = []

    def add(self, name: str, status: str, *, tool: str | None = None,
            document_ids: list[str] | None = None, detail: str = "",
            token_usage: dict[str, int] | None = None) -> None:
        if status not in {"ok", "error", "blocked", "skipped"}:
            raise ValueError("invalid trace status")
        self.steps.append({"name": name, "status": status, "tool": tool,
                           "document_ids": document_ids or [],
                           "detail": redact(detail)[:500],
                           "token_usage": token_usage,
                           "elapsed_ms": round((time.monotonic() - self.started) * 1000)})

    def export(self) -> dict:
        return {"trace_id": self.trace_id, "model": self.model,
                "prompt_version": self.prompt_version,
                "duration_ms": round((time.monotonic() - self.started) * 1000),
                "steps": self.steps}
