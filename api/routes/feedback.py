"""
ProofPilot — Human Feedback & Analyst Override Router
------------------------------------------------------
Collects operational feedback from risk analysts and merchants when disputes
are resolved (won/lost) or when analyst decisions override the model.
Appends to outputs/feedback_log.jsonl for continuous learning & model retraining.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.middleware.api_key_auth import verify_api_key_dependency
from utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/feedback",
    tags=["Human Feedback & Continuous Learning"],
    dependencies=[Depends(verify_api_key_dependency)],
)

FEEDBACK_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "outputs" / "feedback_log.jsonl"


class FeedbackSchema(BaseModel):
    dispute_id: str = Field(..., description="Unique dispute identifier")
    analyst_action: Literal["CONTEST", "ACCEPT_LOSS", "OVERRIDE_CONTEST", "OVERRIDE_ACCEPT"] = Field(
        ..., description="Action taken by human risk analyst"
    )
    actual_outcome: Literal["won", "lost", "pending"] = Field(
        default="pending", description="Actual chargeback resolution outcome if known"
    )
    model_recommendation: str | None = Field(default=None, description="Original model recommendation")
    notes: str | None = Field(default=None, description="Analyst comments, reasoning, or evidence notes")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp of feedback entry",
    )


@router.post("", status_code=status.HTTP_201_CREATED)
def submit_feedback(feedback: FeedbackSchema) -> dict[str, Any]:
    """
    Record analyst feedback, manual overrides, or final dispute resolution outcomes.
    Appends entry to feedback_log.jsonl for continuous audit & MLOps retraining pipelines.
    """
    entry = feedback.model_dump()
    try:
        FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Recorded analyst feedback for dispute {feedback.dispute_id}: {feedback.analyst_action}")
        return {
            "status": "success",
            "message": f"Feedback successfully logged for dispute {feedback.dispute_id}.",
            "dispute_id": feedback.dispute_id,
            "logged_at": entry["timestamp"],
        }
    except Exception as exc:
        logger.error(f"Failed to write feedback for {feedback.dispute_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(exc)}")


@router.get("/summary")
def get_feedback_summary() -> dict[str, Any]:
    """
    Retrieve statistics on analyst feedback, override frequency, and win rates on feedback disputes.
    """
    if not FEEDBACK_LOG_PATH.exists():
        return {
            "total_feedback_entries": 0,
            "overrides_count": 0,
            "actions_breakdown": {},
            "outcomes_breakdown": {},
        }

    entries = []
    try:
        with FEEDBACK_LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line.strip()))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error reading feedback log: {str(exc)}")

    actions: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    overrides = 0

    for e in entries:
        act = e.get("analyst_action", "UNKNOWN")
        actions[act] = actions.get(act, 0) + 1
        if "OVERRIDE" in act:
            overrides += 1
        out = e.get("actual_outcome", "pending")
        outcomes[out] = outcomes.get(out, 0) + 1

    return {
        "total_feedback_entries": len(entries),
        "overrides_count": overrides,
        "actions_breakdown": actions,
        "outcomes_breakdown": outcomes,
    }
