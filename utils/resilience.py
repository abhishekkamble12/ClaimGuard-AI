"""
ProofPilot — Resilience, Fault Tolerance & Circuit Breaker Engine
-----------------------------------------------------------------
Provides robust error recovery for external API calls (Gemini/OpenAI/Razorpay):
1. Exponential backoff retry with full jitter.
2. Thread-safe 3-state Circuit Breaker (CLOSED -> OPEN -> HALF-OPEN).
3. Timeout execution wrapper.
4. ResilientCallExecutor with automatic fallback execution.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from enum import Enum
from typing import Any, Callable, Generic, Optional, Sequence, Tuple, Type, TypeVar

logger = logging.getLogger("proofpilot.resilience")

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenError(Exception):
    """Raised when an operation is attempted while the circuit breaker is OPEN."""
    pass


class ExecutionTimeoutError(Exception):
    """Raised when an operation exceeds its configured execution timeout."""
    pass


class CircuitBreaker:
    """
    Thread-safe 3-state Circuit Breaker pattern implementation.
    
    - CLOSED: Normal operation. Errors increment failure count. If failure count >= threshold, moves to OPEN.
    - OPEN: Requests fail immediately with CircuitBreakerOpenError until recovery_timeout expires.
    - HALF_OPEN: A trial request is permitted. If successful, resets to CLOSED. If fails, returns to OPEN.
    """

    def __init__(
        self,
        name: str = "default_circuit",
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_timeout = max(0.1, recovery_timeout)

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._evaluate_state_locked()
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def _evaluate_state_locked(self) -> None:
        """Evaluate whether an OPEN circuit should transition to HALF_OPEN after timeout."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                logger.info(
                    "CircuitBreaker[%s]: Recovery timeout elapsed (%.1fs). Transitioning OPEN -> HALF_OPEN",
                    self.name,
                    elapsed,
                )
                self._state = CircuitState.HALF_OPEN

    def can_execute(self) -> bool:
        """Check if call is allowed without recording state changes."""
        with self._lock:
            self._evaluate_state_locked()
            return self._state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        """Record a successful invocation, resetting failure counts and closing the circuit."""
        with self._lock:
            if self._state != CircuitState.CLOSED:
                logger.info("CircuitBreaker[%s]: Success recorded. Resetting circuit to CLOSED.", self.name)
            self._state = CircuitState.CLOSED
            self._failure_count = 0

    def record_failure(self, error: Exception | None = None) -> None:
        """Record a failed invocation, incrementing counters and tripping the circuit if needed."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    "CircuitBreaker[%s]: Trial call failed during HALF_OPEN. Tripping back to OPEN. Error: %s",
                    self.name,
                    error,
                )
                self._state = CircuitState.OPEN
            elif self._failure_count >= self.failure_threshold:
                logger.warning(
                    "CircuitBreaker[%s]: Failure threshold (%d) reached. Tripping to OPEN. Error: %s",
                    self.name,
                    self.failure_threshold,
                    error,
                )
                self._state = CircuitState.OPEN


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    jitter: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that applies exponential backoff with full jitter to sync or async callables.
    
    Formula for delay at attempt i:
        raw_delay = min(max_delay, initial_delay * (backoff_factor ** i))
        delay = uniform(0, raw_delay) if jitter else raw_delay
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                attempt = 0
                while True:
                    try:
                        return await fn(*args, **kwargs)
                    except retryable_exceptions as exc:
                        attempt += 1
                        if attempt > max_retries:
                            logger.error(
                                "Function %s failed after %d retries. Final error: %s",
                                fn.__name__,
                                max_retries,
                                exc,
                            )
                            raise exc

                        raw_delay = min(max_delay, initial_delay * (backoff_factor ** (attempt - 1)))
                        delay = random.uniform(0, raw_delay) if jitter else raw_delay
                        logger.warning(
                            "Retry %d/%d for %s after %.3fs due to %s: %s",
                            attempt,
                            max_retries,
                            fn.__name__,
                            delay,
                            type(exc).__name__,
                            exc,
                        )
                        await asyncio.sleep(delay)
            return async_wrapper  # type: ignore

        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                attempt = 0
                while True:
                    try:
                        return fn(*args, **kwargs)
                    except retryable_exceptions as exc:
                        attempt += 1
                        if attempt > max_retries:
                            logger.error(
                                "Function %s failed after %d retries. Final error: %s",
                                fn.__name__,
                                max_retries,
                                exc,
                            )
                            raise exc

                        raw_delay = min(max_delay, initial_delay * (backoff_factor ** (attempt - 1)))
                        delay = random.uniform(0, raw_delay) if jitter else raw_delay
                        logger.warning(
                            "Retry %d/%d for %s after %.3fs due to %s: %s",
                            attempt,
                            max_retries,
                            fn.__name__,
                            delay,
                            type(exc).__name__,
                            exc,
                        )
                        time.sleep(delay)
            return sync_wrapper

    return decorator


def execute_with_timeout(
    fn: Callable[..., T],
    timeout_seconds: float,
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Execute a synchronous callable with an enforcement timeout in a thread pool.
    Raises ExecutionTimeoutError if execution exceeds timeout_seconds.
    """
    if timeout_seconds <= 0:
        return fn(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future: Future[T] = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as exc:
            raise ExecutionTimeoutError(
                f"Call to {getattr(fn, '__name__', str(fn))} timed out after {timeout_seconds}s"
            ) from exc


class ResilientCallExecutor:
    """
    High-level orchestration executor combining:
    - Circuit Breaker check
    - Timeout enforcement
    - Exponential backoff retry
    - Safe fallback invocation
    """

    def __init__(
        self,
        circuit_breaker: CircuitBreaker | None = None,
        max_retries: int = 2,
        initial_delay: float = 0.2,
        backoff_factor: float = 2.0,
        max_delay: float = 5.0,
        timeout_seconds: float = 10.0,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ) -> None:
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.timeout_seconds = timeout_seconds
        self.retryable_exceptions = retryable_exceptions

    def execute(
        self,
        primary_fn: Callable[..., T],
        fallback_fn: Optional[Callable[..., T]] = None,
        fn_args: tuple[Any, ...] = (),
        fn_kwargs: dict[str, Any] | None = None,
        fallback_args: tuple[Any, ...] = (),
        fallback_kwargs: dict[str, Any] | None = None,
    ) -> T:
        """
        Execute primary function under circuit breaker, timeout, and retry protection.
        If all attempts fail or the circuit is OPEN, gracefully invoke fallback_fn.
        """
        fn_kwargs = fn_kwargs or {}
        fallback_kwargs = fallback_kwargs or {}

        # 1. Circuit breaker gate check
        if not self.circuit_breaker.can_execute():
            logger.warning(
                "Circuit breaker '%s' is OPEN. Bypassing primary and routing directly to fallback.",
                self.circuit_breaker.name,
            )
            if fallback_fn is not None:
                return fallback_fn(*fallback_args, **fallback_kwargs)
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.circuit_breaker.name}' is OPEN and no fallback was provided."
            )

        # 2. Retry loop with timeout execution
        attempt = 0
        last_exception: Exception | None = None

        while attempt <= self.max_retries:
            try:
                if self.timeout_seconds > 0:
                    result = execute_with_timeout(
                        primary_fn,
                        self.timeout_seconds,
                        *fn_args,
                        **fn_kwargs,
                    )
                else:
                    result = primary_fn(*fn_args, **fn_kwargs)

                # Record success on circuit breaker
                self.circuit_breaker.record_success()
                return result

            except self.retryable_exceptions as exc:
                last_exception = exc
                attempt += 1

                if attempt <= self.max_retries:
                    raw_delay = min(self.max_delay, self.initial_delay * (self.backoff_factor ** (attempt - 1)))
                    delay = random.uniform(0, raw_delay)
                    logger.warning(
                        "ResilientCallExecutor[%s] attempt %d/%d failed with %s: %s. Retrying in %.3fs...",
                        self.circuit_breaker.name,
                        attempt,
                        self.max_retries,
                        type(exc).__name__,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        # 3. All retries failed: record failure to circuit breaker
        self.circuit_breaker.record_failure(last_exception)

        # 4. Safe fallback invocation
        if fallback_fn is not None:
            logger.info(
                "ResilientCallExecutor[%s]: Primary call failed. Executing safe fallback.",
                self.circuit_breaker.name,
            )
            try:
                return fallback_fn(*fallback_args, **fallback_kwargs)
            except Exception as fallback_exc:
                logger.error(
                    "ResilientCallExecutor[%s]: Fallback execution also failed: %s",
                    self.circuit_breaker.name,
                    fallback_exc,
                )
                raise fallback_exc

        # If no fallback was provided, re-raise the last exception
        assert last_exception is not None
        raise last_exception
