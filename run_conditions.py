"""
Run-condition policy and state for worker execution.

Keeps stop/continue decisions separate from worker orchestration logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass(frozen=True)
class WorkerRunConditions:
    """Immutable run limits and timing configuration."""

    limit: int | None = None
    max_runtime_seconds: float | None = None
    poll_seconds: float = 5.0
    backoff_seconds: float | None = None


@dataclass
class WorkerRunController:
    """Tracks worker progress and evaluates stop/sleep behavior."""

    conditions: WorkerRunConditions
    started_monotonic: float = field(default_factory=time.monotonic)
    total_processed: int = 0

    def remaining_runtime_seconds(self) -> float | None:
        if self.conditions.max_runtime_seconds is None:
            return None

        elapsed = time.monotonic() - self.started_monotonic
        return max(self.conditions.max_runtime_seconds - elapsed, 0.0)

    def stop_reason(self) -> str | None:
        if self.conditions.limit is not None and self.total_processed >= self.conditions.limit:
            return "limit_reached"

        remaining_runtime = self.remaining_runtime_seconds()
        if remaining_runtime is not None and remaining_runtime <= 0:
            return "runtime_reached"

        return None

    def should_stop(self) -> bool:
        return self.stop_reason() is not None

    def record_processed(self, count: int = 1) -> None:
        self.total_processed += count

    def next_batch_limit(self, default_batch_size: int) -> int:
        if self.conditions.limit is None:
            return default_batch_size

        remaining = max(self.conditions.limit - self.total_processed, 0)
        return min(default_batch_size, remaining)

    def sleep_seconds(self, *, idle: bool) -> float:
        configured_sleep = self.conditions.poll_seconds
        if idle and self.conditions.backoff_seconds is not None:
            configured_sleep = self.conditions.backoff_seconds

        configured_sleep = max(configured_sleep, 0.0)

        remaining_runtime = self.remaining_runtime_seconds()
        if remaining_runtime is None:
            return configured_sleep

        return min(configured_sleep, remaining_runtime)
