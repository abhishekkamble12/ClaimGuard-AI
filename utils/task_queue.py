"""
ProofPilot — Asynchronous In-Memory Task & Scoring Job Queue
------------------------------------------------------------
Provides high-throughput asynchronous job queueing for dispute scoring,
batch packet evaluation, and background LLM synthesis.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger("proofpilot.task_queue")


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Job:
    """Represents a discrete unit of asynchronous scoring work."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Any = None
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: Any = None
    error_message: str | None = None
    is_batch: bool = False
    batch_size: int = 1
    handler: Optional[Callable[..., Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "is_batch": self.is_batch,
            "batch_size": self.batch_size,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "latency_ms": round((self.completed_at - self.started_at) * 1000.0, 2)
            if (self.completed_at and self.started_at)
            else None,
            "has_result": self.result is not None,
            "error_message": self.error_message,
        }


class InMemoryJobQueue:
    """
    Asyncio-backed in-memory worker queue for parallel scoring & batch processing.
    """

    def __init__(
        self,
        default_handler: Optional[Callable[..., Any]] = None,
        max_queue_size: int = 10000,
    ) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue_size)
        self._jobs: dict[str, Job] = {}
        self._default_handler = default_handler
        self._workers: list[asyncio.Task[None]] = []
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    def get_job(self, job_id: str) -> Job | None:
        """Retrieve job descriptor by UUID."""
        return self._jobs.get(job_id)

    def get_job_status(self, job_id: str) -> JobStatus | None:
        """Retrieve current lifecycle status for a job."""
        job = self._jobs.get(job_id)
        return job.status if job else None

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        """List summary info for recently submitted jobs."""
        return [job.to_dict() for job in list(self._jobs.values())[-limit:]]

    async def submit_job(
        self,
        payload: Any,
        handler: Optional[Callable[..., Any]] = None,
    ) -> str:
        """
        Submit a single scoring payload to the queue.
        Returns unique job_id UUID.
        """
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            payload=payload,
            status=JobStatus.QUEUED,
            is_batch=False,
            batch_size=1,
            handler=handler or self._default_handler,
        )
        self._jobs[job_id] = job
        await self._queue.put(job_id)
        logger.debug("Submitted single job %s to queue (pending: %d)", job_id, self._queue.qsize())
        return job_id

    async def submit_batch(
        self,
        batch_payload: list[Any],
        handler: Optional[Callable[..., Any]] = None,
    ) -> str:
        """
        Submit a collection/batch of cases as a single orchestrated job.
        Returns unique batch job_id UUID.
        """
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            payload=batch_payload,
            status=JobStatus.QUEUED,
            is_batch=True,
            batch_size=len(batch_payload),
            handler=handler or self._default_handler,
        )
        self._jobs[job_id] = job
        await self._queue.put(job_id)
        logger.info(
            "Submitted batch job %s with %d cases to queue (pending: %d)",
            job_id,
            len(batch_payload),
            self._queue.qsize(),
        )
        return job_id

    async def start_workers(self, worker_count: int = 2) -> None:
        """Start background worker consumer tasks."""
        if self._is_running:
            return

        self._is_running = True
        self._workers = [
            asyncio.create_task(self._worker_loop(i + 1), name=f"proofpilot-worker-{i+1}")
            for i in range(worker_count)
        ]
        logger.info("Started %d background task queue workers.", worker_count)

    async def stop_workers(self) -> None:
        """Gracefully stop and cancel worker loops."""
        if not self._is_running:
            return

        self._is_running = False
        for worker in self._workers:
            worker.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("Stopped all background task queue workers.")

    async def wait_until_complete(self, timeout: float | None = None) -> None:
        """Block until all items in the queue have been fully processed."""
        if timeout:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        else:
            await self._queue.join()

    async def _worker_loop(self, worker_idx: int) -> None:
        """Continuous consumer loop for processing queue jobs."""
        logger.debug("Worker %d loop initialized.", worker_idx)
        while self._is_running:
            try:
                job_id = await self._queue.get()
            except asyncio.CancelledError:
                break

            job = self._jobs.get(job_id)
            if not job:
                self._queue.task_done()
                continue

            job.status = JobStatus.PROCESSING
            job.started_at = time.time()
            handler = job.handler or self._default_handler

            try:
                if handler is None:
                    # Echo / passthrough payload if no handler is specified
                    job.result = job.payload
                else:
                    if inspect.iscoroutinefunction(handler):
                        job.result = await handler(job.payload)
                    else:
                        # Run sync handler in default loop executor to avoid blocking event loop
                        loop = asyncio.get_running_loop()
                        job.result = await loop.run_in_executor(None, handler, job.payload)

                job.status = JobStatus.COMPLETED
            except Exception as exc:
                logger.error("Error processing job %s in worker %d: %s", job_id, worker_idx, exc, exc_info=True)
                job.status = JobStatus.FAILED
                job.error_message = str(exc)
            finally:
                job.completed_at = time.time()
                self._queue.task_done()
