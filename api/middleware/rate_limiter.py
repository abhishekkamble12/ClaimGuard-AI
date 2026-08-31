"""
ProofPilot — In-Memory Token Bucket Rate Limiter
-------------------------------------------------
Protects scoring microservice from abusive request surges.
"""

import time
from collections import defaultdict
from fastapi import HTTPException, Request


class SimpleRateLimiter:
    """
    Token bucket rate limiter keyed by client IP or merchant account.
    """

    def __init__(self, requests_per_minute: int = 120):
        self.rate = requests_per_minute
        self.capacity = requests_per_minute
        self.tokens: dict[str, float] = defaultdict(lambda: float(self.capacity))
        self.last_check: dict[str, float] = defaultdict(time.time)

    def check(self, key: str) -> bool:
        now = time.time()
        elapsed = now - self.last_check[key]
        self.last_check[key] = now

        # Add tokens accumulated over elapsed time
        self.tokens[key] = min(
            float(self.capacity),
            self.tokens[key] + elapsed * (self.rate / 60.0),
        )

        if self.tokens[key] >= 1.0:
            self.tokens[key] -= 1.0
            return True
        return False


_global_limiter = SimpleRateLimiter(requests_per_minute=120)


async def rate_limit_middleware(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown_client"
    if not _global_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please throttle dispute scoring requests.")
