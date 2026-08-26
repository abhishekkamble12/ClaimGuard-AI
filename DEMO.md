# ClaimGuard AI Demo Script

## 1. Start the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 2. Walk through one case

Select `DSP-0009` in the sidebar. Show the merchant, AMEX-style reason code,
amount, Razorpay-aligned order ID, payment ID, payment method, and payment date.

Before the update, the case should show:

- Completeness: 55%
- Decision: Request more evidence
- Main gap: Authentication signal
- Drafting: blocked

Select `Authentication Signal` under **Add or strengthen evidence**, then choose
**Apply Evidence Update**. The case should re-score to:

- Completeness: 80%
- Confidence: 78%
- Decision: Auto-draft response
- Drafting: deterministic response available

Explain that the system does not draft merely because a dispute exists. It
requires both a high evidence score and sufficient confidence.

## 3. Show evaluation

Use the Metrics split selector to show the all-data and held-out test views.
Point out that the benchmark is synthetic and label-derived, while the edge-case
tests exercise defensive behavior for ambiguous or invalid inputs.

## 4. Architecture pitch

1. Reason-code JSON defines the evidence checklist and weights.
2. Evidence documents are classified as present, weak, or missing.
3. Weighted completeness and confidence are calculated.
4. A simple routing gate chooses auto-draft, evidence collection, or human review.
5. The response template runs only after the auto-draft gate passes.

## 5. Responsible automation message

ClaimGuard AI is a pre-submission risk control for merchant dispute operations.
Its primary safety behavior is abstention: weak or ambiguous evidence is sent
back for collection or human review instead of being turned into a confident,
unsupported response.