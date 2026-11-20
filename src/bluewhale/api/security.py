"""API key auth and a per-key sliding-window rate limiter."""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request, status


class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            window = self._hits[key]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.per_minute:
                return False
            window.append(now)
            return True


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> str:
    settings = request.app.state.services.settings
    keys: list[str] = settings.api_keys
    if not keys:
        if settings.is_local:
            principal = "local"
        else:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "API keys not configured")
    elif not x_api_key or not any(hmac.compare_digest(x_api_key, k) for k in keys):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key")
    else:
        principal = f"key:{x_api_key[:4]}"
    if not request.app.state.limiter.check(principal):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
    return principal
