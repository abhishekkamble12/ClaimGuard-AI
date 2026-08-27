import copy
import json
import unittest
from pathlib import Path

from scoring.scorer import infer_evidence_statuses, load_reason_code_config, score_dispute


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "outputs" / "synthetic_chargeback_dataset.json"
CONFIG_PATH = ROOT / "config" / "reason_codes"


def first_case_and_config():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return dataset["cases"][0], load_reason_code_config(CONFIG_PATH)


class EdgeCaseTests(unittest.TestCase):
    def test_missing_evidence_routes_to_human_review(self):
        case, config = first_case_and_config()
        case = copy.deepcopy(case)
        case["evidence_documents"] = {}

        result = score_dispute(case, config)

        self.assertTrue(
            all(
                detail["status"] == "missing"
                for detail in result["evidence_elements"].values()
            )
        )
        self.assertEqual(result["routing_decision"], "human_review")
        self.assertEqual(len(result["missing_evidence"]), 5)

    def test_weak_evidence_is_not_treated_as_present(self):
        case, config = first_case_and_config()
        required = config[case["reason_category"]]["required_evidence"]
        statuses = infer_evidence_statuses(
            required,
            {evidence_id: "Partially available; lacks direct confirmation." for evidence_id in required},
        )

        self.assertTrue(all(status == "weak" for status in statuses.values()))

    def test_misleading_evidence_is_missing(self):
        case, config = first_case_and_config()
        required = config[case["reason_category"]]["required_evidence"]
        statuses = infer_evidence_statuses(
            required,
            {evidence_id: "This document is not related to this transaction." for evidence_id in required},
        )

        self.assertTrue(all(status == "missing" for status in statuses.values()))

    def test_unknown_reason_code_is_rejected(self):
        case, config = first_case_and_config()
        case = copy.deepcopy(case)
        case["reason_category"] = "unknown_reason_code"

        with self.assertRaises(ValueError):
            score_dispute(case, config)

    def test_empty_and_whitespace_documents_are_missing(self):
        case, config = first_case_and_config()
        required = config[case["reason_category"]]["required_evidence"]
        statuses = infer_evidence_statuses(
            required,
            {evidence_id: "   " for evidence_id in required},
        )

        self.assertTrue(all(status == "missing" for status in statuses.values()))


if __name__ == "__main__":
    unittest.main()