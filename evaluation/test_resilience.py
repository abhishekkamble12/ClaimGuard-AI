"""
ProofPilot — Resilience & Circuit Breaker Unit Tests
----------------------------------------------------
Run:
    python -m unittest evaluation.test_resilience
"""

import time
import unittest
from utils.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    ExecutionTimeoutError,
    ResilientCallExecutor,
    execute_with_timeout,
    retry_with_backoff,
)


class TestResilienceEngine(unittest.TestCase):

    def test_circuit_breaker_transitions(self):
        cb = CircuitBreaker(name="test_cb", failure_threshold=2, recovery_timeout=0.2)
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertTrue(cb.can_execute())

        # 1st failure
        cb.record_failure(ValueError("Error 1"))
        self.assertEqual(cb.state, CircuitState.CLOSED)

        # 2nd failure trips circuit to OPEN
        cb.record_failure(ValueError("Error 2"))
        self.assertEqual(cb.state, CircuitState.OPEN)
        self.assertFalse(cb.can_execute())

        # Wait for recovery timeout
        time.sleep(0.25)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        self.assertTrue(cb.can_execute())

        # Successful call resets circuit to CLOSED
        cb.record_success()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertEqual(cb.failure_count, 0)

    def test_retry_with_backoff_success_after_retries(self):
        attempts = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01, backoff_factor=1.5, jitter=False)
        def flaky_api_call():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("Temporary network hiccup")
            return "SUCCESS_DATA"

        result = flaky_api_call()
        self.assertEqual(result, "SUCCESS_DATA")
        self.assertEqual(attempts, 3)

    def test_retry_with_backoff_exhaustion(self):
        attempts = 0

        @retry_with_backoff(max_retries=2, initial_delay=0.01, jitter=False)
        def consistently_failing():
            nonlocal attempts
            attempts += 1
            raise TimeoutError("Endpoint down")

        with self.assertRaises(TimeoutError):
            consistently_failing()

        self.assertEqual(attempts, 3)  # 1 initial + 2 retries

    def test_execute_with_timeout(self):
        def fast_func():
            return "ok"

        def slow_func():
            time.sleep(0.3)
            return "slow_ok"

        self.assertEqual(execute_with_timeout(fast_func, timeout_seconds=1.0), "ok")

        with self.assertRaises(ExecutionTimeoutError):
            execute_with_timeout(slow_func, timeout_seconds=0.1)

    def test_resilient_executor_with_fallback(self):
        cb = CircuitBreaker(name="executor_cb", failure_threshold=2, recovery_timeout=0.5)
        executor = ResilientCallExecutor(
            circuit_breaker=cb,
            max_retries=1,
            initial_delay=0.01,
            timeout_seconds=0.5,
        )

        def failing_primary():
            raise RuntimeError("Primary LLM unavailable")

        def safe_fallback():
            return "FALLBACK_LETTER_DRAFT"

        # Primary fails -> Retries -> Invokes fallback
        result = executor.execute(
            primary_fn=failing_primary,
            fallback_fn=safe_fallback,
        )
        self.assertEqual(result, "FALLBACK_LETTER_DRAFT")

    def test_resilient_executor_circuit_open_direct_fallback(self):
        cb = CircuitBreaker(name="open_cb", failure_threshold=1, recovery_timeout=10.0)
        cb.record_failure(Exception("Manual trip"))
        self.assertEqual(cb.state, CircuitState.OPEN)

        executor = ResilientCallExecutor(circuit_breaker=cb, max_retries=0)

        primary_called = False

        def primary_fn():
            nonlocal primary_called
            primary_called = True
            return "SHOULD_NOT_REACH"

        def fallback_fn():
            return "FAST_FALLBACK"

        result = executor.execute(primary_fn=primary_fn, fallback_fn=fallback_fn)
        self.assertFalse(primary_called)
        self.assertEqual(result, "FAST_FALLBACK")


if __name__ == "__main__":
    unittest.main()
