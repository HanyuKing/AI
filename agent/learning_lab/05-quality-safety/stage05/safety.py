"""对不可信工具/检索文本做标记；不能替代隔离和权限控制。"""

import re

INJECTION_PATTERNS = (r"ignore (all )?(previous|prior) instructions", r"忽略.{0,8}(指令|规则)",
                      r"reveal (the )?(system prompt|secret)", r"泄露.{0,8}(提示词|密钥)")


def validate_user_input(text: str, max_chars: int = 2000) -> dict:
    if not text.strip():
        return {"accepted": False, "reason": "empty_input"}
    if len(text) > max_chars:
        return {"accepted": False, "reason": "input_too_long"}
    return {"accepted": True, "normalized": text.strip()}


def inspect_untrusted_text(text: str) -> dict:
    flagged = any(re.search(pattern, text, re.IGNORECASE) for pattern in INJECTION_PATTERNS)
    return {"trusted": not flagged, "text": text, "warning": "untrusted content" if flagged else None}
