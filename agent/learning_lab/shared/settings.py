"""小型环境配置对象：集中读取、校验配置，避免业务代码散落 os.getenv。"""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    app_env: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000
    request_timeout_seconds: float = 10.0
    max_concurrent_model_calls: int = 8

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            app_env=os.getenv("APP_ENV", "development"),
            host=os.getenv("APP_HOST", "127.0.0.1"),
            port=int(os.getenv("APP_PORT", "8000")),
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10")),
            max_concurrent_model_calls=int(os.getenv("MAX_CONCURRENT_MODEL_CALLS", "8")),
        )
        if not 1 <= settings.port <= 65535:
            raise ValueError("APP_PORT must be between 1 and 65535")
        if settings.request_timeout_seconds <= 0:
            raise ValueError("REQUEST_TIMEOUT_SECONDS must be positive")
        if settings.max_concurrent_model_calls < 1:
            raise ValueError("MAX_CONCURRENT_MODEL_CALLS must be at least 1")
        return settings
