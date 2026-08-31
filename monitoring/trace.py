"""
ProofPilot — Inference & LLM Execution Tracer
-----------------------------------------------
Provides production telemetry, latency percentile tracking, token consumption,
cost estimation, and per-dispute audit tracing for all LLM and ML inference calls.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("proofpilot.trace")

# Estimated cost per 1k tokens for standard dispute reasoning models (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"input_per_1k": 0.000075, "output_per_1k": 0.000300},
    "gemini-1.5-flash": {"input_per_1k": 0.000075, "output_per_1k": 0.000300},
    "gemini-1.5-pro": {"input_per_1k": 0.001250, "output_per_1k": 0.005000},
    "gpt-4o-mini": {"input_per_1k": 0.000150, "output_per_1k": 0.000600},
    "gpt-4o": {"input_per_1k": 0.002500, "output_per_1k": 0.010000},
    "local-rules-fallback": {"input_per_1k": 0.0, "output_per_1k": 0.0},
}


def calculate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost in USD based on model pricing."""
    pricing = MODEL_PRICING.get(model, {"input_per_1k": 0.0001, "output_per_1k": 0.0004})
    cost = (input_tokens / 1000.0) * pricing["input_per_1k"] + (output_tokens / 1000.0) * pricing["output_per_1k"]
    return round(cost, 6)


@dataclass
class TraceSpan:
    """Represents a single instrumented inference execution span."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    dispute_id: str | None = None
    operation_name: str = "llm_inference"
    model: str = "gemini-2.5-flash"
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    status: str = "RUNNING"  # RUNNING | SUCCESS | FAILED
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(
        self,
        status: str = "SUCCESS",
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        error_message: str | None = None,
        cost_usd: float | None = None,
    ) -> "TraceSpan":
        """Close span, calculate latency, token aggregates, and estimated cost."""
        self.end_time = time.time()
        self.latency_ms = round((self.end_time - self.start_time) * 1000.0, 2)
        self.status = status

        if input_tokens is not None:
            self.input_tokens = input_tokens
        if output_tokens is not None:
            self.output_tokens = output_tokens
        if error_message is not None:
            self.error_message = str(error_message)

        if cost_usd is not None:
            self.cost_usd = cost_usd
        else:
            self.cost_usd = calculate_cost_usd(self.model, self.input_tokens, self.output_tokens)

        return self

    def __enter__(self) -> "TraceSpan":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_val is not None:
            self.finish(status="FAILED", error_message=str(exc_val))
        elif self.status == "RUNNING":
            self.finish(status="SUCCESS")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InferenceTracer:
    """
    Thread-safe telemetry and trace aggregator for ML / LLM operations.
    Maintains per-dispute execution history, latency percentiles, and cost metrics.
    """

    _instance: Optional["InferenceTracer"] = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        self._spans: list[TraceSpan] = []
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "InferenceTracer":
        """Get or initialize the global thread-safe singleton instance."""
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def record_span(self, span: TraceSpan) -> TraceSpan:
        """Add a completed or active span to the tracer."""
        with self._lock:
            self._spans.append(span)
        return span

    def start_span(
        self,
        operation_name: str = "llm_inference",
        dispute_id: str | None = None,
        model: str = "gemini-2.5-flash",
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan:
        """Create, register, and return a new TraceSpan."""
        span = TraceSpan(
            dispute_id=dispute_id,
            operation_name=operation_name,
            model=model,
            metadata=metadata or {},
        )
        with self._lock:
            self._spans.append(span)
        return span

    def get_all_spans(self) -> list[TraceSpan]:
        """Return a snapshot list of all recorded spans."""
        with self._lock:
            return list(self._spans)

    def get_dispute_traces(self, dispute_id: str) -> list[dict[str, Any]]:
        """Retrieve all execution spans associated with a specific dispute ID."""
        with self._lock:
            return [span.to_dict() for span in self._spans if span.dispute_id == dispute_id]

    def reset(self) -> None:
        """Clear all in-memory spans (useful for tests)."""
        with self._lock:
            self._spans.clear()

    @staticmethod
    def _compute_percentile(sorted_values: list[float], percentile: float) -> float:
        """Compute the given percentile from an already sorted list of floats."""
        if not sorted_values:
            return 0.0
        k = (len(sorted_values) - 1) * (percentile / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return round(sorted_values[int(k)], 2)
        d0 = sorted_values[int(f)] * (c - k)
        d1 = sorted_values[int(c)] * (k - f)
        return round(d0 + d1, 2)

    def get_summary_metrics(self) -> dict[str, Any]:
        """
        Aggregate tracer metrics:
        - Latency percentiles (p50, p90, p95, p99, min, max, avg)
        - Total token counts (input, output, aggregate)
        - Cumulative and average cost (USD)
        - Success / failure rates
        - Grouped breakdown by model
        """
        with self._lock:
            spans_copy = list(self._spans)

        total_calls = len(spans_copy)
        if total_calls == 0:
            return {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "error_rate": 0.0,
                "latency_ms": {
                    "p50": 0.0,
                    "p90": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                    "avg": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                },
                "tokens": {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0,
                    "avg_tokens_per_call": 0.0,
                },
                "financials": {
                    "total_cost_usd": 0.0,
                    "avg_cost_per_call_usd": 0.0,
                },
                "by_model": {},
            }

        successful = [s for s in spans_copy if s.status == "SUCCESS"]
        failed = [s for s in spans_copy if s.status == "FAILED"]
        latencies = sorted([s.latency_ms for s in spans_copy])

        total_input_tokens = sum(s.input_tokens for s in spans_copy)
        total_output_tokens = sum(s.output_tokens for s in spans_copy)
        total_tokens = total_input_tokens + total_output_tokens
        total_cost = sum(s.cost_usd for s in spans_copy)

        # Model breakdown
        by_model: dict[str, dict[str, Any]] = {}
        for s in spans_copy:
            if s.model not in by_model:
                by_model[s.model] = {"calls": 0, "tokens": 0, "cost_usd": 0.0, "latency_sum": 0.0}
            by_model[s.model]["calls"] += 1
            by_model[s.model]["tokens"] += (s.input_tokens + s.output_tokens)
            by_model[s.model]["cost_usd"] = round(by_model[s.model]["cost_usd"] + s.cost_usd, 6)
            by_model[s.model]["latency_sum"] += s.latency_ms

        for m_data in by_model.values():
            if m_data["calls"] > 0:
                m_data["avg_latency_ms"] = round(m_data["latency_sum"] / m_data["calls"], 2)
            del m_data["latency_sum"]

        return {
            "total_calls": total_calls,
            "successful_calls": len(successful),
            "failed_calls": len(failed),
            "error_rate": round(len(failed) / total_calls, 4),
            "latency_ms": {
                "p50": self._compute_percentile(latencies, 50),
                "p90": self._compute_percentile(latencies, 90),
                "p95": self._compute_percentile(latencies, 95),
                "p99": self._compute_percentile(latencies, 99),
                "avg": round(sum(latencies) / total_calls, 2),
                "min": round(latencies[0], 2),
                "max": round(latencies[-1], 2),
            },
            "tokens": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "avg_tokens_per_call": round(total_tokens / total_calls, 1),
            },
            "financials": {
                "total_cost_usd": round(total_cost, 6),
                "avg_cost_per_call_usd": round(total_cost / total_calls, 6),
            },
            "by_model": by_model,
        }

    def export_dispute_traces(
        self,
        dispute_id: str,
        output_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Export all trace spans for a given dispute ID to a JSON file or list."""
        traces = self.get_dispute_traces(dispute_id)
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(traces, indent=2), encoding="utf-8")
            logger.info("Exported %d traces for dispute %s to %s", len(traces), dispute_id, path)
        return traces

    def export_all(self, output_path: str | Path) -> dict[str, Any]:
        """Export full telemetry summary and span log to JSON."""
        data = {
            "summary": self.get_summary_metrics(),
            "spans": [s.to_dict() for s in self.get_all_spans()],
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Exported full telemetry to %s", path)
        return data


def get_global_tracer() -> InferenceTracer:
    """Convenience accessor for global InferenceTracer singleton."""
    return InferenceTracer.get_instance()


def get_inference_tracer() -> InferenceTracer:
    """Convenience accessor for global InferenceTracer singleton."""
    return InferenceTracer.get_instance()


def trace_span(
    operation_name: str = "llm_inference",
    dispute_id: str | None = None,
    model: str = "gemini-2.5-flash",
    metadata: dict[str, Any] | None = None,
) -> TraceSpan:
    """Helper to start and return an instrumented TraceSpan (can be used as context manager)."""
    return get_global_tracer().start_span(
        operation_name=operation_name,
        dispute_id=dispute_id,
        model=model,
        metadata=metadata,
    )

