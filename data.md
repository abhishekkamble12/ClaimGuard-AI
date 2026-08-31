# ClaimGuard AI Dataset Generation Guide

## Best Dataset Strategy

For the Razorpay AI Buildathon, generate a synthetic but rule-grounded chargeback dataset.

The dataset should prove that ClaimGuard AI can:

- Detect required evidence for a reason code
- Mark evidence as present, weak, or missing
- Predict whether the case is ready for response drafting
- Route risky cases to human review
- Report precision, recall, abstention rate, and false-positive cost

The records use Razorpay-aligned merchant and payment metadata for demo realism,
but they are synthetic and are not sourced from Razorpay or a card network.

## Recommended Dataset Size

Use this split:

```text
60 total disputes
41 training/tuning disputes
19 held-out test disputes
```

This is large enough for a credible hackathon evaluation and small enough to inspect manually.

## Recommended Categories

Use 5 dispute categories:

| Category | Why It Is Useful |
| --- | --- |
| goods_not_received | Easy demo with delivery proof |
| product_not_as_described | Tests policy and communication evidence |
| refund_not_processed | Tests refund timeline and policy evidence |
| unauthorized_fraud | Tests device, IP, and authentication evidence |
| duplicate_charge | Tests transaction comparison evidence |

## Label Format

Each required evidence item should be labeled as:

```text
present
weak
missing
```

These labels become ground truth for ML and evaluation.

## How To Generate

Run:

```bash
python data_generator.py --cases 60 --test-ratio 0.33 --seed 42 --out outputs/synthetic_chargeback_dataset.json
```

For a bigger dataset:

```bash
python data_generator.py --cases 300 --test-ratio 0.2 --out outputs/synthetic_chargeback_dataset_300.json
```

## How To Use For ML Training

Use each dispute as one training row.

Suggested input features:

- reason_code
- reason_category
- amount
- payment_method
- evidence item statuses
- number of missing evidence items
- number of weak evidence items
- completeness score
- confidence score

Suggested prediction targets:

- expected_route
- expected_outcome
- expected_completeness_score

For a first ML model, train a classifier for:

```text
expected_route:
- auto_draft_response
- request_more_evidence
- human_review
```

## Important Note

Do not claim this is real chargeback data. In the pitch, say:

> We built a 60-case synthetic chargeback benchmark with reason-code-specific evidence labels and held-out evaluation.

That is honest, credible, and aligned with the Buildathon requirement for measured precision and recall.

For a stronger evaluation claim, treat the manually designed tests in
`evaluation/test_edge_cases.py` as behavioral safeguards, not benchmark training
data. They cover missing, weak, misleading, empty, and unknown-reason inputs.
