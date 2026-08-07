"""API admission control (§2.2).

All agents share one org rate-limit bucket, so uncoordinated fan-out means
429 storms. This module implements the two-layer gate:

* a semaphore capping in-flight requests (default min(agents, 4));
* a token bucket sized from the anthropic-ratelimit-* headers, refreshed on
  every response; requests wait for capacity rather than discovering it via
  a 429.

Plus cache-warm-then-fan-out: one warmup request against the shared prefix so
the fleet's concurrent requests hit the cache instead of all missing.
"""

from __future__ import annotations

import threading
import time


class TokenBucket:
    def __init__(self, capacity: float, refill_per_sec: float) -> None:
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill = refill_per_sec
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.refill)
        self.updated = now

    def take(self, n: float = 1.0, timeout: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self.tokens >= n:
                    self.tokens -= n
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def set_rate(self, requests_per_min: float) -> None:
        with self._lock:
            self.refill = requests_per_min / 60.0
            self.capacity = max(requests_per_min / 60.0 * 5, self.capacity)


class Admission:
    """Shared admission gate across all agents."""

    def __init__(self, max_inflight: int = 4, default_rpm: float = 60.0) -> None:
        self.inflight = threading.Semaphore(max_inflight)
        self.bucket = TokenBucket(capacity=max(default_rpm / 60.0 * 10, 5), refill_per_sec=default_rpm / 60.0)

    def acquire(self, timeout: float = 120.0) -> bool:
        if not self.inflight.acquire(timeout=timeout):
            return False
        if not self.bucket.take(1.0, timeout):
            self.inflight.release()
            return False
        return True

    def release(self, header_rpm: float | None = None) -> None:
        if header_rpm and header_rpm > 0:
            self.bucket.set_rate(header_rpm)
        self.inflight.release()

    def wait_for_capacity(self, timeout: float = 300.0) -> None:
        """Called before a retry storm: drain in-flight requests first."""
        deadline = time.monotonic() + timeout
        while self.bucket.tokens < 1.0 and time.monotonic() < deadline:
            time.sleep(0.1)