"""
ProofPilot — Dataset Cases Router
-----------------------------------
Provides endpoints for retrieving benchmark dataset cases for the frontend Case Selector.
"""

import json
from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/dataset", tags=["Dataset Cases"])

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = ROOT / "outputs" / "synthetic_chargeback_dataset.json"


def _load_dataset() -> dict[str, Any]:
    if not DATASET_PATH.exists():
        from data_generator import generate_dataset
        ds = generate_dataset(500, 0.25, 42)
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        DATASET_PATH.write_text(json.dumps(ds, indent=2), encoding="utf-8")
        return ds
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@router.get("/cases")
def get_dataset_cases() -> dict[str, Any]:
    """Retrieve all dispute cases from the benchmark dataset."""
    ds = _load_dataset()
    return {
        "dataset_name": ds.get("dataset_name", "proofpilot_synthetic_chargeback_dataset"),
        "total_cases": len(ds.get("cases", [])),
        "cases": ds.get("cases", []),
    }


@router.get("/cases/{dispute_id}")
def get_dataset_case_by_id(dispute_id: str) -> dict[str, Any]:
    """Retrieve a single dispute case by its dispute_id."""
    ds = _load_dataset()
    for case in ds.get("cases", []):
        if case.get("dispute_id") == dispute_id:
            return case
    raise HTTPException(status_code=404, detail=f"Dispute case '{dispute_id}' not found in dataset.")
