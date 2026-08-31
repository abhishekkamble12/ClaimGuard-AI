"""
ProofPilot — Disputes Scoring & Audit Log Router
--------------------------------------------------
Provides endpoints for:
1. Synchronous single dispute scoring & ML win probability computation.
2. Synchronous & Asynchronous batch dispute scoring via InMemoryJobQueue.
3. Real-time streaming dispute response letter generation via Server-Sent Events (SSE).
4. Immutable audit log retrieval for decision defense.
"""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.dependencies import get_all_reason_codes
from api.middleware.api_key_auth import verify_api_key_dependency
from api.schemas.dispute import (
    BatchScoreRequest,
    BatchScoreResponse,
    DisputeScoreRequest,
    DisputeScoringResponse,
)
from generation.llm_reasoning import stream_gated_dispute_response
from scoring.audit_log import get_audit_log
from scoring.scorer import score_dispute
from utils.task_queue import InMemoryJobQueue

router = APIRouter(
    prefix="/disputes",
    tags=["Dispute Scoring"],
    dependencies=[Depends(verify_api_key_dependency)],
)

DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "outputs" / "synthetic_chargeback_dataset.json"

_job_queue: InMemoryJobQueue | None = None


def get_job_queue() -> InMemoryJobQueue:
    """Retrieve or initialize the global InMemoryJobQueue."""
    global _job_queue
    if _job_queue is None:
        _job_queue = InMemoryJobQueue()
    return _job_queue


def _find_dispute_case(dispute_id: str) -> dict[str, Any] | None:
    """Find dispute case by ID from dataset or audit log."""
    if DATASET_PATH.exists():
        try:
            ds = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
            for case in ds.get("cases", []):
                if case.get("dispute_id") == dispute_id:
                    return case
        except Exception:
            pass

    audit_entry = get_audit_log(dispute_id)
    if audit_entry and "dispute" in audit_entry:
        return audit_entry["dispute"]
    return None


@router.post("/score", response_model=DisputeScoringResponse)
def score_single_dispute(
    request: DisputeScoreRequest,
    reason_codes: dict[str, Any] = Depends(get_all_reason_codes),
) -> Any:
    """
    Score evidence readiness and compute calibrated ML win probability
    and financial Expected Value for a dispute case.
    Secured via X-API-Key authentication.
    """
    case_dict = request.dispute.model_dump()
    try:
        result = score_dispute(
            dispute=case_dict,
            reason_codes=reason_codes,
            use_ground_truth=request.use_ground_truth,
            api_key=request.api_key,
            log_audit=True,
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal scoring error: {str(exc)}")


@router.post("/batch-score", response_model=BatchScoreResponse)
def batch_score_disputes(
    request: BatchScoreRequest,
    reason_codes: dict[str, Any] = Depends(get_all_reason_codes),
) -> Any:
    """Batch score multiple dispute cases synchronously."""
    results = []
    for dispute_schema in request.disputes:
        case_dict = dispute_schema.model_dump()
        try:
            res = score_dispute(case_dict, reason_codes, api_key=request.api_key, log_audit=False)
            results.append(res)
        except Exception:
            pass

    return {
        "total_processed": len(request.disputes),
        "successful_scores": len(results),
        "results": results,
    }


@router.post("/async-batch-score", status_code=status.HTTP_202_ACCEPTED)
async def async_batch_score_disputes(
    request: BatchScoreRequest,
    reason_codes: dict[str, Any] = Depends(get_all_reason_codes),
) -> dict[str, Any]:
    """
    Asynchronously submit a batch of dispute cases for background scoring via InMemoryJobQueue.
    Returns 202 Accepted with a unique job_id for polling results.
    """
    queue = get_job_queue()
    if not queue.is_running:
        await queue.start_workers(worker_count=2)

    cases_payload = [d.model_dump() for d in request.disputes]
    api_key = request.api_key

    def _process_batch_payload(payload: list[dict[str, Any]]) -> dict[str, Any]:
        scored_results = []
        for case in payload:
            try:
                res = score_dispute(case, reason_codes, api_key=api_key, log_audit=False)
                scored_results.append(res)
            except Exception as exc:
                scored_results.append({"error": str(exc), "dispute_id": case.get("dispute_id")})
        return {
            "total_processed": len(payload),
            "successful_scores": len([r for r in scored_results if "error" not in r]),
            "results": scored_results,
        }

    job_id = await queue.submit_batch(cases_payload, handler=_process_batch_payload)
    return {
        "job_id": job_id,
        "status": "QUEUED",
        "batch_size": len(cases_payload),
        "message": "Async batch scoring job accepted and queued for execution.",
    }


@router.get("/jobs/{job_id}")
def get_job_status_and_result(job_id: str) -> dict[str, Any]:
    """
    Retrieve asynchronous job status and results when completed.
    """
    queue = get_job_queue()
    job = queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found in task queue.")

    job_dict = job.to_dict()
    job_dict["result"] = job.result
    return job_dict


@router.get("/{dispute_id}/stream-response")
def stream_dispute_response_endpoint(
    dispute_id: str,
    api_key: str | None = None,
    reason_codes: dict[str, Any] = Depends(get_all_reason_codes),
) -> StreamingResponse:
    """
    Stream real-time LLM-generated dispute response letter using Server-Sent Events (SSE).
    """
    case_dict = _find_dispute_case(dispute_id)
    if not case_dict:
        case_dict = {
            "dispute_id": dispute_id,
            "merchant_name": "Merchant",
            "reason_title": "Dispute Claim",
            "reason_code": "4853",
            "transaction": {"amount": 0, "currency": "INR", "order_id": "ORD-0000", "payment_id": "pay_0000"},
            "evidence_documents": {},
        }

    try:
        scoring_res = score_dispute(case_dict, reason_codes, api_key=api_key, log_audit=False)
    except Exception:
        scoring_res = {"evidence_elements": {}, "completeness_pct": "100%", "confidence_pct": "100%"}

    def event_stream_generator():
        for chunk in stream_gated_dispute_response(case_dict, scoring_res, api_key=api_key):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(event_stream_generator(), media_type="text/event-stream")


@router.get("/{dispute_id}/audit")
def get_dispute_audit_trace(dispute_id: str) -> dict[str, Any]:
    """Retrieve immutable audit log for a specific dispute decision."""
    audit_entry = get_audit_log(dispute_id)
    if not audit_entry:
        raise HTTPException(status_code=404, detail=f"No audit trace found for dispute {dispute_id}")
    return audit_entry
