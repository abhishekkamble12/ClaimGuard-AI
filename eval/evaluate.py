"""
Legacy eval.evaluate shim — delegates to evaluation.evaluate
"""
from evaluation.evaluate import *

if __name__ == "__main__":
    from evaluation.evaluate import evaluate_dataset, print_report
    report = evaluate_dataset()
    print_report(report)
