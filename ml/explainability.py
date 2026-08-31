"""
ProofPilot — ML Explainability & XAI Diagnostic Engine
-------------------------------------------------------
Provides four XAI visualization layers for Razorpay risk analysts:

  1. Global SHAP Summary Plot     — feature impact across all training cases
  2. Local SHAP Waterfall         — per-dispute P(Win) explanation (Phase 1)
  3. Evidence Contribution Chart  — weighted readiness decomposition (Phase 2)

All functions fall back to matplotlib-only alternatives if the optional
`shap` library is unavailable, ensuring zero hard dependency failures.
"""

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1A — Global SHAP Summary Plot (existing, unchanged signature)
# ─────────────────────────────────────────────────────────────────────────────

def generate_shap_plot(
    model: Any,
    X_train: np.ndarray,
    feature_names: list[str],
    max_display: int = 7,
) -> plt.Figure:
    """
    Generate SHAP summary feature attribution plot for the win prediction model.
    Shows global average feature impact across all training disputes.
    Returns a matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)

    try:
        import shap

        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_train)

        if isinstance(shap_values, list) and len(shap_values) == 2:
            vals_to_plot = shap_values[1]
        else:
            vals_to_plot = shap_values

        plt.clf()
        shap.summary_plot(
            vals_to_plot,
            X_train,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
            plot_size=(8, 4.5),
        )
        fig = plt.gcf()
        fig.patch.set_facecolor("#ffffff")
        plt.title(
            "SHAP Feature Impact on Dispute Win Probability (Global)",
            fontsize=11,
            pad=12,
            fontweight="bold",
        )
    except Exception:
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        coefs = np.abs(model.coef_[0])[:max_display]
        names = feature_names[:max_display]
        y_pos = np.arange(len(names))
        ax.barh(y_pos, coefs, align="center", color="#3b82f6", edgecolor="none")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |Coefficient| Contribution", fontsize=9)
        ax.set_title(
            "Feature Contribution to Win Probability (Global Coefficients)",
            fontsize=11,
            fontweight="bold",
        )
        fig.tight_layout()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1B — Local SHAP Waterfall (per-dispute P(Win) explanation)
# ─────────────────────────────────────────────────────────────────────────────

def explain_dispute_prediction(
    model: Any,
    X_train: np.ndarray,
    single_instance: np.ndarray,
    feature_names: list[str],
    base_prob: float,
    predicted_prob: float,
) -> plt.Figure:
    """
    Generate a per-dispute SHAP waterfall chart explaining why this specific
    dispute received the P(Win) it did.

    Green bars push P(Win) upward (favorable evidence features).
    Red bars push P(Win) downward (unfavorable features: missing evidence, high count).

    Falls back to a manual signed bar chart using model coefficients × feature values
    if the `shap` library is unavailable.

    Args:
        model:           Trained LogisticRegression (or compatible sklearn model).
        X_train:         Full training feature matrix (for SHAP background).
        single_instance: 2D array shape (1, n_features) for the current dispute.
        feature_names:   Human-readable names for each feature dimension.
        base_prob:       Dataset average P(Win) (SHAP base/expected value).
        predicted_prob:  Model's predicted P(Win) for this dispute.

    Returns:
        matplotlib Figure with the waterfall chart.
    """
    max_display = 10

    try:
        import shap

        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(single_instance)

        # LinearExplainer returns shape (1, n_features)
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) == 2 else shap_values[0][0]
        else:
            sv = shap_values[0]

        explanation = shap.Explanation(
            values=sv,
            base_values=explainer.expected_value if np.isscalar(explainer.expected_value)
                        else explainer.expected_value[1] if len(explainer.expected_value) == 2
                        else explainer.expected_value[0],
            data=single_instance[0],
            feature_names=feature_names,
        )

        plt.clf()
        shap.waterfall_plot(explanation, max_display=max_display, show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 4.5)
        fig.patch.set_facecolor("#ffffff")
        fig.suptitle(
            f"Local XAI — Why P(Win) = {predicted_prob:.0%} for This Dispute",
            fontsize=11,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        return fig

    except Exception:
        # ── Fallback: manual signed contribution bar chart ──
        contributions = model.coef_[0] * single_instance[0]

        # Sort by absolute contribution magnitude, take top max_display
        sorted_idx = np.argsort(np.abs(contributions))[::-1][:max_display]
        sorted_contributions = contributions[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
        colors = ["#22c55e" if v >= 0 else "#ef4444" for v in sorted_contributions]
        y_pos = np.arange(len(sorted_names))
        bars = ax.barh(y_pos, sorted_contributions, align="center", color=colors, edgecolor="none", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(x=0, color="#374151", linewidth=1.0, linestyle="-")
        ax.set_xlabel("Feature Contribution to P(Win) (Weight × Value)", fontsize=9)
        ax.set_title(
            f"Local XAI — P(Win) = {predicted_prob:.0%} | Why This Dispute Scored This Way",
            fontsize=11,
            fontweight="bold",
        )

        # Annotate value labels
        for bar, val in zip(bars, sorted_contributions):
            x_label = val + 0.002 if val >= 0 else val - 0.002
            ha = "left" if val >= 0 else "right"
            ax.text(x_label, bar.get_y() + bar.get_height() / 2,
                    f"{val:+.3f}", va="center", ha=ha, fontsize=8, color="#374151")

        # Legend
        green_patch = mpatches.Patch(color="#22c55e", label="Pushes toward WIN ↑")
        red_patch = mpatches.Patch(color="#ef4444", label="Pushes toward LOSS ↓")
        ax.legend(handles=[green_patch, red_patch], fontsize=8, loc="lower right")
        fig.patch.set_facecolor("#ffffff")
        fig.tight_layout()
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Evidence Contribution Chart (weighted readiness decomposition)
# ─────────────────────────────────────────────────────────────────────────────

def build_evidence_contribution_chart(
    evidence_elements: dict[str, Any],
    completeness_score: float,
    auto_draft_threshold: float = 0.80,
) -> plt.Figure:
    """
    Horizontal bar chart decomposing the total readiness score into individual
    evidence element contributions. Color-coded by evidence status:
      - Green  : present  (full contribution realized)
      - Amber  : weak     (partial contribution)
      - Light grey: missing (zero contribution; full potential shown as outline)

    A vertical dashed line marks the auto-draft threshold (default 80%).
    This makes the exact evidence gap immediately visually obvious.

    Args:
        evidence_elements:    Dict from score_dispute() result["evidence_elements"].
        completeness_score:   Total weighted readiness score (0.0–1.0).
        auto_draft_threshold: The minimum readiness score to pass the auto-draft gate.

    Returns:
        matplotlib Figure.
    """
    STATUS_COLORS = {
        "present": "#22c55e",    # green
        "weak":    "#f59e0b",    # amber
        "missing": "#e5e7eb",    # light grey
        "irrelevant": "#d1d5db", # grey
    }
    STATUS_EDGE = {
        "present": "#16a34a",
        "weak":    "#d97706",
        "missing": "#9ca3af",
        "irrelevant": "#9ca3af",
    }

    # Sort by weight descending
    items = sorted(evidence_elements.items(), key=lambda x: x[1]["weight"], reverse=True)
    names = [k.replace("_", " ").title() for k, _ in items]
    contributions = [v["weighted_contribution"] for _, v in items]
    full_potentials = [v["weight"] for _, v in items]
    statuses = [v["status"] for _, v in items]
    weights = [v["weight"] for _, v in items]

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(items) * 0.55 + 1.2)), dpi=100)

    y_pos = np.arange(len(names))

    # Draw full-potential ghost bars for missing/weak items
    for i, (status, potential) in enumerate(zip(statuses, full_potentials)):
        if status in ("missing", "weak"):
            ax.barh(y_pos[i], potential, align="center", color="#f3f4f6",
                    edgecolor="#d1d5db", linewidth=0.8, height=0.55, linestyle="--", zorder=1)

    # Draw actual contribution bars
    for i, (status, contrib) in enumerate(zip(statuses, contributions)):
        color = STATUS_COLORS.get(status, "#9ca3af")
        edge  = STATUS_EDGE.get(status, "#6b7280")
        ax.barh(y_pos[i], contrib if contrib > 0 else 0.001, align="center",
                color=color, edgecolor=edge, linewidth=0.8, height=0.55, zorder=2)

    # Auto-draft threshold vertical line
    ax.axvline(x=auto_draft_threshold, color="#6366f1", linewidth=1.5,
               linestyle="--", zorder=3, label=f"Auto-Draft Threshold ({auto_draft_threshold:.0%})")
    # Total score marker
    ax.axvline(x=completeness_score, color="#0f172a", linewidth=2.0,
               linestyle="-", zorder=4, label=f"Current Readiness ({completeness_score:.0%})")

    # Labels: weight on left, contribution value + status on bar
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{n}  ({w:.0%})" for n, w in zip(names, weights)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Weighted Score Contribution", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_title(
        f"Evidence Contribution Breakdown — Total Readiness: {completeness_score:.0%}",
        fontsize=11,
        fontweight="bold",
    )

    # Annotate contribution values
    for i, (status, contrib, potential) in enumerate(zip(statuses, contributions, full_potentials)):
        if contrib > 0.015:
            ax.text(contrib - 0.008, y_pos[i], f"{contrib:.0%}",
                    va="center", ha="right", fontsize=8, color="#ffffff", fontweight="bold", zorder=5)
        if status == "missing":
            ax.text(potential + 0.01, y_pos[i], "Missing ❌",
                    va="center", ha="left", fontsize=8, color="#9ca3af", zorder=5)
        elif status == "weak":
            ax.text(contrib + 0.01, y_pos[i], f"Weak ⚠️ (+{potential - contrib:.0%} possible)",
                    va="center", ha="left", fontsize=8, color="#d97706", zorder=5)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#22c55e", edgecolor="#16a34a", label="Present ✅ (full contribution)"),
        mpatches.Patch(color="#f59e0b", edgecolor="#d97706", label="Weak ⚠️ (partial contribution)"),
        mpatches.Patch(color="#e5e7eb", edgecolor="#9ca3af", linestyle="--", label="Missing ❌ (zero — potential shown)"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
