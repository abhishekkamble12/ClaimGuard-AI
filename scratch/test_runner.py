import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    print("=== Running All Unit & Integration Tests ===")
    loader = unittest.defaultTestLoader
    suite_eval = loader.discover(start_dir=str(ROOT / "evaluation"), pattern="test_*.py", top_level_dir=str(ROOT))
    suite_api = loader.discover(start_dir=str(ROOT / "api" / "tests"), pattern="test_*.py", top_level_dir=str(ROOT))
    all_tests = unittest.TestSuite([suite_eval, suite_api])

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(all_tests)
    
    if not result.wasSuccessful():
        print("TESTS FAILED!")
        sys.exit(1)
        
    print("\n=== Running evaluation/evaluate.py ===")
    from evaluation.evaluate import evaluate_dataset, print_report
    metrics = evaluate_dataset()
    print_report(metrics)

    print("\n=== Running evaluation/ragas_eval.py ===")
    from evaluation.ragas_eval import run_ragas_evaluation
    ragas_res = run_ragas_evaluation(split="test")
    print(f"RAGAS Faithfulness: {ragas_res['metrics']['mean_faithfulness']:.1%}")
    print(f"RAGAS Correctness : {ragas_res['metrics']['mean_answer_correctness']:.1%}")
    print(f"RAGAS Recall@5    : {ragas_res['metrics']['mean_evidence_recall_at_5']:.1%}")
    print(f"RAGAS Gap MRR     : {ragas_res['metrics']['mean_reciprocal_rank_gap_mrr']:.3f}")

    print("\n=== Testing Audit Log Rotation CLI ===")
    from scoring.audit_log import prune_audit_logs
    deleted = prune_audit_logs(90)
    print(f"Pruned logs: {deleted}")

    print("\nALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
