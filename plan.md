# ClaimGuard-AI / ProofPilot — 13-Point Technical Improvements Plan

Comprehensive ML pipeline hardening to fix precision/recall issues, improve model rigor, and add production-grade polish for the Razorpay Buildathon.

---

## Current State Summary (from codebase audit)

| Aspect | Current Status | Gap |
|--------|---------------|-----|
| **Stacking** | Weighted average ensemble (GBT+LR+RF) trained on same split, `CalibratedClassifierCV(cv=3)` wraps each base model on training data | ❌ No OOF predictions for meta-learner — potential leakage |
| **Evaluation Split** | Single 375/125 train/test split with 80/20 train/val sub-split | ❌ No K-Fold cross-validation for metrics |
| **Thresholds** | Global EV-optimal threshold via `precision_recall_curve` on val split | ❌ No per-category thresholds |
| **Class Imbalance** | `class_weight={0:1, 1:5}` for LR/RF, `sample_weight` for GBT, `scale_pos_weight=5.0` for XGB | ✅ Already partially handled |
| **Critical Evidence** | Features exist (`Any_Critical_Missing`, `Has_All_Critical_Evidence`) | ❌ No hard-penalty post-hoc rule |
| **Extractor** | LLM (Gemini) primary, rule-based fallback | ⚠️ LLM path exists but prompt is basic |
| **Semantic Matcher** | TF-IDF cosine similarity only | ❌ No sentence embeddings |
| **Monotonic Constraints** | Not set on any tree model | ❌ Missing |
| **Calibration** | Brier score + calibration_curve.png visual | ❌ No numeric ECE |
| **Confidence Intervals** | Bootstrap CI in benchmark.py | ❌ No Wilson CI on headline metrics |
| **Ablation** | Single-split model comparison table | ❌ Not K-Fold validated |
| **Drift Detection** | PSI code exists in `model_monitor.py` | ❌ No live endpoint/dashboard |
| **Test Coverage** | Test files exist but results not reported in README | ❌ No pass rate in README |

---

## Proposed Changes

### P0 — Fundamental Model Fixes (Must-do first)

---

### 1. OOF-Based Stacking (Fix Leakage)

#### [MODIFY] [win_predictor.py](file:///d:/ClaimGuard-AI/ml/win_predictor.py)

**Current:** Base models are fit on `X_train_arr`, then `CalibratedClassifierCV(cv=3)` is fit on the same `X_train_arr`. The ensemble weights are computed on `X_val_arr` using predictions from models that saw `X_train_arr` during both raw fitting and calibration. This creates subtle leakage — the calibrated models' CV folds may overlap with the data used to compute ensemble weights.

**Change:**
- Use `sklearn.model_selection.cross_val_predict(method='predict_proba')` with `StratifiedKFold(5)` on the training set to generate out-of-fold (OOF) probability predictions for each base model
- Use these OOF predictions to learn ensemble weights (instead of val-split predictions from potentially leaked models)
- Add a comment block: `# OOF-based stacking — no train-set leakage`
- Store OOF predictions in the model artifact for reproducibility

```python
from sklearn.model_selection import cross_val_predict, StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Generate OOF predictions for each base model
oof_preds = {}
for name, raw_model in self.raw_models.items():
    oof_preds[name] = cross_val_predict(
        raw_model, self.X_train_arr, self.y_train_arr,
        cv=skf, method='predict_proba'
    )[:, 1]

# Learn weights from OOF predictions (no leakage)
oof_precs = {}
for name, oof_prob in oof_preds.items():
    pred = (oof_prob >= 0.40).astype(int)
    denom = max(int(pred.sum()), 1)
    oof_precs[name] = max(0.01, float((pred & self.y_train_arr).sum()) / denom)
total_prec = sum(oof_precs.values())
self.ensemble_weights = {k: round(v / total_prec, 4) for k, v in oof_precs.items()}

# Then re-fit base models on FULL training set for final deployment
for name, raw_model in self.raw_models.items():
    raw_model.fit(self.X_train_arr, self.y_train_arr)
```

---

### 2. Stratified K-Fold Cross-Validation for Metrics

#### [MODIFY] [evaluate.py](file:///d:/ClaimGuard-AI/evaluation/evaluate.py)

**Current:** Metrics are computed on a single 125-case test split. High variance with small N.

**Change:**
- Add a new function `evaluate_kfold()` that runs `StratifiedKFold(n_splits=5)` on the **entire** 500-case dataset
- In each fold: train the ensemble on 400 cases, evaluate on 100 cases
- Report mean ± std for precision, recall, F1, AUC, Brier, PR-AUC across 5 folds
- Keep the existing `evaluate_dataset()` for single-split evaluation (backward compatible)
- Update the console report to show K-Fold results as the primary headline metrics

```python
def evaluate_kfold(n_splits=5) -> dict:
    """Run StratifiedKFold cross-validation and report mean ± std metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        # Train on fold's training set, evaluate on fold's test set
        # Collect precision, recall, f1, auc, brier, pr_auc per fold
        fold_metrics.append({...})
    # Return mean ± std for each metric
```

#### [MODIFY] [README.md](file:///d:/ClaimGuard-AI/README.md)

- Update headline metrics to show K-Fold mean ± std format (e.g., "Precision: 55.2% ± 4.1%")
- Add note: "All metrics reported as 5-Fold Stratified CV mean ± std"

---

### 3. Per-Category Threshold Optimization

#### [MODIFY] [win_predictor.py](file:///d:/ClaimGuard-AI/ml/win_predictor.py)

**Current:** Single global `ev_optimal_threshold` used for all categories.

**Change:**
- In `find_ev_optimal_threshold()`, after finding the global threshold, compute per-category thresholds:
  - Group validation cases by `reason_category`
  - For each category with ≥5 cases, run `precision_recall_curve` and find cost-minimizing threshold: `FP_count × 500 + FN_count × avg_amount`
  - Store per-category thresholds in `self.category_thresholds: dict[str, float]`

```python
self.category_thresholds = {}
for cat in set(c.get("reason_category") for c in val_cases_meta):
    cat_mask = [c.get("reason_category") == cat for c in val_cases_meta]
    if sum(cat_mask) < 5:
        continue
    cat_probs = val_probs[cat_mask]
    cat_y = y_val[cat_mask]
    cat_amounts = amounts[cat_mask]
    # Find cost-minimizing threshold for this category
    best_t, best_cost = self._find_cost_minimizing_threshold(cat_probs, cat_y, cat_amounts)
    self.category_thresholds[cat] = best_t
```

#### [MODIFY] `recommend_action()` in [win_predictor.py](file:///d:/ClaimGuard-AI/ml/win_predictor.py)

- Look up `self.category_thresholds.get(category, self.ev_optimal_threshold)` instead of using global threshold

#### [NEW] [config/category_thresholds.json](file:///d:/ClaimGuard-AI/config/category_thresholds.json)

- Persist per-category thresholds alongside the model artifact for transparency

---

### 4. Class Imbalance — Verify & Document

#### [MODIFY] [win_predictor.py](file:///d:/ClaimGuard-AI/ml/win_predictor.py)

**Current Status:** ✅ Already partially implemented:
- LR: `class_weight={0: 1.0, 1: 5.0}` ✅
- RF: `class_weight={0: 1.0, 1: 5.0}` ✅
- GBT: `sample_weight` with 5x for positives ✅
- XGB: `scale_pos_weight=5.0` ✅

**Change:**
- Compute `scale_pos_weight` dynamically from actual data: `n_negative / n_positive`
- For GBT, use `class_weight='balanced'` equivalent via dynamic `sample_weight` calculation
- Add explicit comment documenting the rationale

```python
n_pos = self.y_train_arr.sum()
n_neg = len(self.y_train_arr) - n_pos
dynamic_scale = round(n_neg / max(n_pos, 1), 2)
# Use dynamic_scale instead of hardcoded 5.0
```

---

### 5. Critical Evidence Missing — Hard Penalty Feature

#### [MODIFY] [feature_engineering.py](file:///d:/ClaimGuard-AI/ml/feature_engineering.py)

**Current:** Features `Any_Critical_Missing` and `Critical_Evidence_Missing_Count` exist but are just input features — the model can learn to ignore them.

**Change:**
- Add a post-hoc hard cap in `predict_win_probability()` and `recommend_action()`:

#### [MODIFY] [win_predictor.py](file:///d:/ClaimGuard-AI/ml/win_predictor.py)

```python
def predict_win_probability(self, dispute, scoring_result):
    # ... existing ensemble prediction ...
    ensemble_prob = ...  # existing code
    
    # Hard-penalty: if critical evidence (weight >= 0.20) is missing, cap probability
    evidence_elements = scoring_result.get("evidence_elements", {})
    critical_missing = any(
        detail.get("status") == "missing" and float(detail.get("weight", 0)) >= 0.20
        for detail in evidence_elements.values()
        if isinstance(detail, dict)
    )
    if critical_missing:
        ensemble_prob = min(ensemble_prob, 0.15)  # hard cap
    
    return round(ensemble_prob, 4)
```

---

### P1 — Extraction & Feature Improvements

---

### 6. LLM-Backed Extractor Enhancement

#### [MODIFY] [extractor.py](file:///d:/ClaimGuard-AI/extraction/extractor.py)

**Current:** LLM path exists (Gemini 2.5 Flash) but the prompt is a simple status classification. Rule-based fallback is keyword matching.

**Change:**
- Enhance the LLM prompt to evaluate **legal sufficiency**, not just keyword presence:

```python
prompt = f"""You are an expert payment dispute analyst.
Classify each evidence document based on LEGAL SUFFICIENCY for a chargeback rebuttal, not just keyword presence.

Rules:
- "present": Document directly proves the merchant's position with legally admissible evidence
- "weak": Document exists but has gaps (e.g., delivery tracking without signature, partial refund proof)
- "missing": No document provided, or document is irrelevant/unrelated to this specific transaction
- "irrelevant": Document is for a different transaction or completely unrelated

Consider: Is this evidence strong enough to win a chargeback arbitration?
"""
```

- Add Groq/Llama 3.3 70B as an alternative LLM path (fast, free tier) for extraction
- In `evaluate.py`, add an ablation row comparing rule-based vs LLM extraction F1

#### [MODIFY] [evaluate.py](file:///d:/ClaimGuard-AI/evaluation/evaluate.py)

- Add extraction ablation: run evaluation twice (once with LLM extraction, once with rule-based) and report comparison table

---

### 7. TF-IDF → Sentence Embeddings

#### [MODIFY] [semantic_matcher.py](file:///d:/ClaimGuard-AI/ml/semantic_matcher.py)

**Current:** Pure TF-IDF + cosine similarity. Fails on paraphrasing and adversarial text.

**Change:**
- Add `sentence-transformers` as optional dependency
- Use `all-MiniLM-L6-v2` (local, fast, no API cost) as primary matcher
- Fall back to TF-IDF if `sentence-transformers` is not installed

```python
try:
    from sentence_transformers import SentenceTransformer
    _st_model = SentenceTransformer('all-MiniLM-L6-v2')
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

def compute_semantic_relevance(document_text, evidence_description):
    if _HAS_ST:
        embeddings = _st_model.encode([document_text, evidence_description])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return round(float(sim), 4)
    # Fallback to TF-IDF
    ...
```

#### [MODIFY] [requirements.txt](file:///d:/ClaimGuard-AI/requirements.txt)

- Add `sentence-transformers>=2.2.0` (optional, but recommended)

---

### 8. Monotonic Constraints on GBT/XGBoost

#### [MODIFY] [win_predictor.py](file:///d:/ClaimGuard-AI/ml/win_predictor.py)

**Current:** No monotonic constraints on any tree model.

**Change:**
- Add monotonic constraints to GBT and XGBoost ensuring that features like `completeness_score`, `confidence`, `evidence_count` always have a non-negative relationship with win probability

```python
# Build monotonic constraints vector matching feature order
# +1 = increasing, -1 = decreasing, 0 = unconstrained
feature_names = self.feature_extractor.get_feature_names()
mono_map = {
    "Completeness Score": 1,
    "Confidence Score": 1,
    "Missing Count": -1,
    "Weak Count": -1,
    "Has Critical Evidence": 1,
    "Semantic Relevance Mean": 1,
    "Evidence Completeness Ratio": 1,
    "Critical Evidence Present Count": 1,
    "Has All Critical Evidence": 1,
    "Any Critical Missing": -1,
    "Critical Evidence Missing Count": -1,
}
mono_constraints = [mono_map.get(f, 0) for f in feature_names]

# GBT (sklearn doesn't support monotonic_cst natively in GBC, use XGBoost/LightGBM)
# XGBoost:
raw_xgb = XGBClassifier(
    ...,
    monotone_constraints=tuple(mono_constraints),
)
```

> [!IMPORTANT]
> sklearn's `GradientBoostingClassifier` does NOT support `monotone_constraints`. Two options:
> 1. Replace sklearn GBT with `HistGradientBoostingClassifier` (supports `monotonic_cst` since sklearn 1.3+)
> 2. Apply constraints only to XGBoost (already available)
>
> **Recommendation:** Apply to XGBoost only (already the primary tree model), since sklearn GBT is a secondary model in the ensemble.

---

### 9. ECE (Expected Calibration Error) — Numeric Calibration Proof

#### [MODIFY] [calibration_plot.py](file:///d:/ClaimGuard-AI/evaluation/calibration_plot.py)

**Current:** Only visual calibration curve + Brier score annotation.

**Change:**
- Add `expected_calibration_error()` function
- Print ECE alongside Brier in the chart annotation
- Save ECE to `metrics.json`

```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return round(ece, 4)
```

#### [MODIFY] [evaluate.py](file:///d:/ClaimGuard-AI/evaluation/evaluate.py)

- Compute and include ECE in the metrics output alongside Brier score

#### [MODIFY] [README.md](file:///d:/ClaimGuard-AI/README.md)

- Add ECE to the metrics table: "ECE: 0.0X (10-bin)"

---

### P2 — Robustness & Production Polish

---

### 10. Confidence Intervals on Headline Metrics (Wilson CI)

#### [MODIFY] [evaluate.py](file:///d:/ClaimGuard-AI/evaluation/evaluate.py)

**Current:** Bootstrap CI exists in `benchmark.py` but not in the main evaluation output.

**Change:**
- Add Wilson score CIs for precision, recall on the test split using `scipy.stats.binomtest`

```python
from scipy.stats import binomtest

def wilson_ci(successes, trials, confidence=0.95):
    if trials == 0:
        return (0.0, 0.0)
    result = binomtest(successes, trials)
    ci = result.proportion_ci(confidence_level=confidence, method='wilson')
    return (round(ci.low, 4), round(ci.high, 4))

# Example: Precision CI
prec_ci = wilson_ci(dec_tp, dec_tp + dec_fp)
# Recall CI
rec_ci = wilson_ci(dec_tp, dec_tp + dec_fn)
```

- Include CIs in `metrics.json` and `print_report()`

---

### 11. Ablation Table — K-Fold Validated

#### [MODIFY] [evaluate.py](file:///d:/ClaimGuard-AI/evaluation/evaluate.py) or [benchmark.py](file:///d:/ClaimGuard-AI/evaluation/benchmark.py)

**Current:** Ablation table (LR vs RF vs GBT vs Ensemble) is from single split.

**Change:**
- Run each model through 5-Fold CV and report mean ± std per model
- This naturally falls out of the K-Fold evaluation in Point #2 — extend it to report per-model metrics per fold

---

### 12. Live Drift Detection Dashboard

#### [NEW] API endpoint in [main.py](file:///d:/ClaimGuard-AI/api/main.py) or new route file

**Current:** PSI code exists in `model_monitor.py` but no endpoint exposes it.

**Change:**
- Add a `/api/v1/drift` endpoint that returns current PSI scores per feature
- Add a `/api/v1/drift/alert` endpoint that returns whether any feature exceeds PSI threshold
- Optionally add a Streamlit dashboard panel showing PSI status

```python
@router.get("/drift")
def get_drift_report():
    monitor = get_drift_monitor()
    reports = monitor.evaluate_drift()
    should_retrain, drifted = monitor.should_retrain()
    return {
        "should_retrain": should_retrain,
        "drifted_features": drifted,
        "feature_reports": [r.to_dict() for r in reports],
    }
```

---

### 13. Test Suite Coverage — Run & Report

#### [MODIFY] [README.md](file:///d:/ClaimGuard-AI/README.md)

**Current:** Test files exist (`test_edge_cases.py`, `test_resilience.py`, `test_ensemble_and_api.py`, etc.) but no pass rates reported.

**Change:**
- Run all test suites with `pytest` and record results
- Add a "Test Coverage" section to README: e.g., "42/42 tests passing"
- Consider adding a GitHub Actions CI workflow (`.github/workflows/test.yml`) that runs tests on every push

---

## Open Questions

> [!IMPORTANT]
> **Q1: HistGradientBoosting vs sklearn GBT for monotonic constraints?**
> sklearn's `GradientBoostingClassifier` doesn't support monotonic constraints. Should we:
> - (a) Replace it with `HistGradientBoostingClassifier` (supports `monotonic_cst`), or
> - (b) Only apply constraints to XGBoost and leave GBT unchanged?
> Option (b) is simpler and lower risk since XGB is already the primary tree model.

> [!IMPORTANT]
> **Q2: Groq/Llama for extraction — API key availability?**
> Point #6 suggests adding Groq/Llama 3.3 70B as an alternative LLM extractor. Do you have a Groq API key configured? If not, we'll enhance only the existing Gemini extraction prompt.

> [!IMPORTANT]
> **Q3: `sentence-transformers` model download — acceptable?**
> `all-MiniLM-L6-v2` is ~80MB. Is it acceptable to add this as a required dependency, or should it remain optional with TF-IDF fallback?

---

## Verification Plan

### Automated Tests
```bash
# Run all existing tests
python -m pytest evaluation/ -v --tb=short

# Run K-Fold evaluation
python evaluation/evaluate.py --split all --retrain

# Generate calibration plot with ECE
python evaluation/calibration_plot.py

# Run benchmark with CIs
python evaluation/benchmark.py
```

### Manual Verification
1. **0% recall categories**: After per-category thresholds (#3) + class weights (#4) + hard penalty (#5), verify that `unauthorized_fraud` and `product_not_as_described` show non-zero recall
2. **Metrics stability**: K-Fold std should be < 10pp for precision/recall
3. **ECE value**: Should be < 0.10 for reasonable calibration
4. **Drift endpoint**: Hit `/api/v1/drift` and verify JSON response with PSI scores
5. **Test suite**: All tests pass with `pytest`

### Execution Priority
```
Day 1-2: Points #1-5 (P0 — fundamental model fixes)
Day 3:   Points #6-9 (P1 — extraction & features)  
Day 4:   Points #10-13 (P2 — polish & robustness)
```
