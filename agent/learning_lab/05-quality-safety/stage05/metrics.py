"""离线汇总延迟与模型成本的纯函数，方便单测和 CI 使用。"""


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_latency(samples_ms: list[float]) -> dict:
    return {"count": len(samples_ms), "p50_ms": round(percentile(samples_ms, 50), 2),
            "p95_ms": round(percentile(samples_ms, 95), 2),
            "max_ms": round(max(samples_ms), 2) if samples_ms else 0}


def estimate_cost(input_tokens: int, output_tokens: int,
                  input_usd_per_million: float, output_usd_per_million: float) -> float:
    if min(input_tokens, output_tokens, input_usd_per_million, output_usd_per_million) < 0:
        raise ValueError("token counts and rates must not be negative")
    return (input_tokens * input_usd_per_million + output_tokens * output_usd_per_million) / 1_000_000
