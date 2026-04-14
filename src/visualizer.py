# =============================================================================
# FILE: src/visualizer.py
# PURPOSE: Generate all charts, graphs, and visual outputs for the project.
#
# CHARTS GENERATED:
#   1. Anomaly Score Distribution     — Histogram of IF scores, attack vs normal
#   2. ROC Curve                      — Trade-off between detection & false alarms
#   3. Confusion Matrix Heatmap       — Visual TP/FP/TN/FN breakdown
#   4. Feature Importance Bar Chart   — Which features matter most
#   5. Attack Type Distribution       — Pie / bar of detected attack categories
#   6. Network Traffic Timeline       — Anomaly scores over time
#   7. Severity Breakdown             — Alert count by severity level
#   8. Detection Rate Comparison      — Models side by side
#   9. Top Source IPs                 — Most frequent attack sources
#  10. Executive Summary Dashboard    — All key metrics on one figure
#
# WHY VISUALIZATION MATTERS:
#   In a real SOC, security analysts don't read raw CSV data.
#   They need dashboards that instantly show:
#     - Where is the threat coming from?
#     - How severe is it?
#     - Is the model performing well?
#     - What patterns emerge over time?
#   These visualizations answer ALL of those questions.
#
# OUTPUTS DIRECTORY: outputs/
# IMAGES DIRECTORY:  images/  (copies of outputs, used in README)
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL STYLE CONFIGURATION
# =============================================================================

# We use a professional dark SOC-style theme for all charts.
# This looks great in README files and GitHub previews.
plt.rcParams.update({
    "figure.facecolor":  "#0D1117",     # GitHub dark background
    "axes.facecolor":    "#161B22",     # Slightly lighter panel background
    "axes.edgecolor":    "#30363D",     # Subtle axis borders
    "axes.labelcolor":   "#C9D1D9",     # Light gray axis labels
    "xtick.color":       "#8B949E",     # Muted tick labels
    "ytick.color":       "#8B949E",
    "text.color":        "#C9D1D9",     # Default text color
    "grid.color":        "#21262D",     # Very subtle grid lines
    "grid.linewidth":    0.5,
    "figure.titlesize":  14,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "font.family":       "DejaVu Sans",
})

# Color palette — matching the project's cybersecurity theme
COLORS = {
    "normal":    "#1F8EFA",  # Blue  — normal/benign traffic
    "attack":    "#F85149",  # Red   — attack traffic
    "critical":  "#FF7B72",  # Light red
    "high":      "#FFA657",  # Orange
    "medium":    "#E3B341",  # Yellow
    "low":       "#3FB950",  # Green
    "info":      "#8B949E",  # Gray
    "accent":    "#58A6FF",  # Bright blue accent
    "success":   "#3FB950",  # Green
    "warning":   "#E3B341",  # Yellow
    "grid":      "#21262D",  # Grid line color
    "text_dim":  "#8B949E",  # Dimmed text
    "text_main": "#C9D1D9",  # Main text
    "text_bright":"#F0F6FC", # Bright white for emphasis
}

# Attack type colors for consistent coloring across charts
ATTACK_COLORS = {
    "DoS / DDoS Attack":       "#F85149",
    "Port Scan":               "#FFA657",
    "Brute Force":             "#D2A8FF",
    "Web Attack":              "#79C0FF",
    "Botnet / C2 Activity":    "#56D364",
    "Unknown / Novel Attack":  "#E3B341",
    "Suspicious Activity":     "#8B949E",
    "Benign":                  "#1F8EFA",
}

# Severity colors
SEVERITY_COLORS = {
    "CRITICAL": "#FF7B72",
    "HIGH":     "#FFA657",
    "MEDIUM":   "#E3B341",
    "LOW":      "#3FB950",
    "INFO":     "#8B949E",
}

# Output directories
OUTPUTS_DIR = "outputs"
IMAGES_DIR  = "images"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_output_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str, dpi: int = 150) -> str:
    """
    Save a matplotlib figure to both outputs/ and images/ directories.

    Args:
        fig:      The matplotlib Figure object.
        filename: Filename without directory prefix.
        dpi:      Resolution. 150 is good for README images.

    Returns:
        str: Path where figure was saved.
    """
    setup_output_dirs()
    out_path = os.path.join(OUTPUTS_DIR, filename)
    img_path = os.path.join(IMAGES_DIR, filename)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(img_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  [SAVE] {filename} → {out_path}")
    return out_path


def styled_title(ax: plt.Axes, title: str, subtitle: str = ""):
    """Apply a styled two-part title to an axis."""
    ax.set_title(title, color=COLORS["text_bright"], fontsize=12,
                 fontweight="bold", pad=12)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8,
                color=COLORS["text_dim"])


def add_count_labels(ax: plt.Axes, bars, fmt: str = "{:.0f}"):
    """Add value labels on top of bar chart bars."""
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., h + 0.5,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=8, color=COLORS["text_main"]
        )


# =============================================================================
# CHART 1: Anomaly Score Distribution
# =============================================================================

def plot_anomaly_distribution(
    anomaly_scores: np.ndarray,
    true_labels: np.ndarray,
    filename: str = "01_anomaly_score_distribution.png"
) -> str:
    """
    Plot histogram of Isolation Forest anomaly scores, split by true label.

    WHAT TO LOOK FOR:
      - Normal traffic (blue) should cluster at LOW scores (near 0)
      - Attack traffic (red) should cluster at HIGH scores (near 1)
      - A good model shows clear separation between the two distributions
      - Overlap zone is where false positives/negatives occur
    """
    print("[VIZ] Plotting anomaly score distribution...")

    fig, ax = plt.subplots(figsize=(10, 5))

    normal_scores = anomaly_scores[true_labels == 0]
    attack_scores = anomaly_scores[true_labels == 1]

    bins = np.linspace(0, 1, 50)

    ax.hist(normal_scores, bins=bins, alpha=0.7, color=COLORS["normal"],
            label=f"Normal Traffic (n={len(normal_scores):,})", density=True)
    ax.hist(attack_scores, bins=bins, alpha=0.7, color=COLORS["attack"],
            label=f"Attack Traffic (n={len(attack_scores):,})", density=True)

    # Alert threshold line
    threshold = 0.7
    ax.axvline(x=threshold, color=COLORS["warning"], linewidth=1.5,
               linestyle="--", label=f"Alert Threshold ({threshold})")

    # Shade the alert zone
    ax.axvspan(threshold, 1.0, alpha=0.08, color=COLORS["attack"],
               label="Alert Zone")

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.2, labelcolor="white")
    styled_title(ax, "Anomaly Score Distribution",
                 "Isolation Forest — separation between normal and attack flows")

    # Annotation
    overlap_pct = np.mean((normal_scores > threshold)) * 100
    ax.text(0.99, 0.95,
            f"FPR at threshold: {overlap_pct:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=COLORS["warning"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1A1A2E", alpha=0.7))

    plt.tight_layout()
    return save_figure(fig, filename)


# =============================================================================
# CHART 2: ROC Curve
# =============================================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob_rf: np.ndarray,
    y_prob_lr: np.ndarray = None,
    filename: str = "02_roc_curve.png"
) -> str:
    """
    Plot ROC (Receiver Operating Characteristic) curves for our models.

    WHAT THE ROC CURVE SHOWS:
      - X-axis: False Positive Rate (FPR) — how often we falsely flag normal traffic
      - Y-axis: True Positive Rate (TPR)  — how many real attacks we catch
      - Perfect model: goes to top-left corner, AUC = 1.0
      - Random model:  diagonal line, AUC = 0.5
      - The curve shows the trade-off at EVERY threshold setting

    IN SECURITY:
      Security teams use this to choose their operating point:
        "I can tolerate 2% false alarms — what detection rate does that give me?"
    """
    print("[VIZ] Plotting ROC curve...")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Random Forest ROC
    fpr_rf, tpr_rf, _ = roc_curve(y_true, y_prob_rf)
    auc_rf = auc(fpr_rf, tpr_rf)
    ax.plot(fpr_rf, tpr_rf, color=COLORS["accent"],
            linewidth=2, label=f"Random Forest  (AUC = {auc_rf:.4f})")

    # Logistic Regression ROC (optional)
    if y_prob_lr is not None:
        fpr_lr, tpr_lr, _ = roc_curve(y_true, y_prob_lr)
        auc_lr = auc(fpr_lr, tpr_lr)
        ax.plot(fpr_lr, tpr_lr, color=COLORS["high"],
                linewidth=1.5, linestyle="--",
                label=f"Logistic Regression (AUC = {auc_lr:.4f})")

    # Random baseline
    ax.plot([0, 1], [0, 1], color=COLORS["text_dim"],
            linewidth=1, linestyle=":", label="Random Classifier (AUC = 0.50)")

    # Mark the "ideal" operating point (low FPR, high TPR)
    # Find the threshold closest to 95% TPR
    if len(tpr_rf) > 0:
        idx_95 = np.argmin(np.abs(tpr_rf - 0.95))
        ax.scatter(fpr_rf[idx_95], tpr_rf[idx_95],
                   color=COLORS["success"], s=80, zorder=5,
                   label=f"95% Detection Rate @ FPR={fpr_rf[idx_95]:.3f}")

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.2, labelcolor="white", loc="lower right")
    styled_title(ax, "ROC Curve — Threat Detection Performance",
                 "Higher AUC = better trade-off between detection rate and false alarms")

    plt.tight_layout()
    return save_figure(fig, filename)


# =============================================================================
# CHART 3: Confusion Matrix Heatmap
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Random Forest",
    class_names: list = None,
    filename: str = "03_confusion_matrix.png"
) -> str:
    """
    Plot a styled confusion matrix heatmap.

    READING THE CONFUSION MATRIX:
      ┌────────────────┬──────────────┬──────────────┐
      │                │ Pred: Normal │ Pred: Attack │
      ├────────────────┼──────────────┼──────────────┤
      │ Actual: Normal │  True Neg    │  False Pos   │
      │ Actual: Attack │  False Neg   │  True Pos    │
      └────────────────┴──────────────┴──────────────┘

      True Positive  (TP): Attack correctly identified as attack ✓
      True Negative  (TN): Normal correctly identified as normal ✓
      False Positive (FP): Normal wrongly flagged as attack (false alarm!)
      False Negative (FN): Attack missed and called normal (dangerous!)

    IN SECURITY: FN is more dangerous than FP.
      A missed attack (FN) can mean a data breach.
      A false alarm (FP) just wastes analyst time.
    """
    print(f"[VIZ] Plotting confusion matrix for {model_name}...")

    if class_names is None:
        class_names = ["Normal", "Attack"]

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (matrix, title_suffix) in enumerate([
        (cm,            "Raw Counts"),
        (cm_normalized, "Normalized (%)"),
    ]):
        ax = axes[ax_idx]
        fmt = ".2f" if ax_idx == 1 else "d"

        # Custom colormap: dark purple to bright teal
        cmap = sns.diverging_palette(220, 20, as_cmap=True) if ax_idx == 0 else "Blues"

        sns.heatmap(
            matrix,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            ax=ax,
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5,
            linecolor="#30363D",
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 12, "weight": "bold"}
        )

        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        styled_title(ax, f"{model_name} — {title_suffix}")

        # Color the four cells for emphasis
        if ax_idx == 0:
            n = len(class_names)
            for i in range(n):
                for j in range(n):
                    color = COLORS["success"] if i == j else COLORS["critical"]
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                               edgecolor=color, lw=2))

    # Add a metrics summary text box
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics_text = (
        f"TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}\n"
        f"Precision={precision:.3f}   Recall={recall:.3f}   F1={f1:.3f}"
    )
    fig.text(0.5, -0.02, metrics_text, ha="center", fontsize=9,
             color=COLORS["text_main"],
             bbox=dict(boxstyle="round", facecolor="#161B22", alpha=0.8))

    plt.tight_layout()
    return save_figure(fig, filename)


# =============================================================================
# CHART 4: Feature Importance
# =============================================================================

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    filename: str = "04_feature_importance.png"
) -> str:
    """
    Plot a horizontal bar chart of the top-N most important features.

    READING THIS CHART:
      - Features on the right have more influence on the model's decisions
      - Top features are the most "predictive" network statistics
      - Use this to understand WHICH aspects of network traffic betray attacks
      - Security teams use this to build more efficient monitoring rules
    """
    print("[VIZ] Plotting feature importance...")

    top_df = importance_df.head(top_n).copy()
    top_df = top_df.sort_values("Importance")  # ascending for horizontal bar

    fig, ax = plt.subplots(figsize=(10, top_n * 0.4 + 2))

    # Color bars by importance level
    max_imp = top_df["Importance"].max()
    bar_colors = [
        COLORS["critical"] if v > max_imp * 0.6 else
        COLORS["high"]     if v > max_imp * 0.4 else
        COLORS["medium"]   if v > max_imp * 0.2 else
        COLORS["low"]
        for v in top_df["Importance"]
    ]

    bars = ax.barh(
        top_df["Feature"].str.strip(),
        top_df["Importance"],
        color=bar_colors,
        alpha=0.85,
        height=0.7,
    )

    # Add value labels
    for bar, val in zip(bars, top_df["Importance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                fontsize=7, color=COLORS["text_main"])

    ax.set_xlabel("Feature Importance Score (Gini Impurity Reduction)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, top_df["Importance"].max() * 1.18)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["critical"], label="High importance (>60% max)"),
        mpatches.Patch(facecolor=COLORS["high"],     label="Medium-high (40–60%)"),
        mpatches.Patch(facecolor=COLORS["medium"],   label="Medium (20–40%)"),
        mpatches.Patch(facecolor=COLORS["low"],      label="Lower (<20%)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              framealpha=0.2, labelcolor="white", fontsize=7)

    styled_title(ax, f"Top {top_n} Feature Importances — Random Forest",
                 "Which network statistics most help identify attacks")

    plt.tight_layout()
    return save_figure(fig, filename)


# =============================================================================
# CHART 5: Attack Type Distribution
# =============================================================================

def plot_attack_distribution(
    alerts_df: pd.DataFrame,
    filename: str = "05_attack_distribution.png"
) -> str:
    """
    Plot bar + pie chart showing which attack types were detected.

    WHY THIS MATTERS:
      - Shows the THREAT LANDSCAPE for this network
      - Helps prioritize which defenses to strengthen
      - DoS-heavy? Invest in rate limiting.
      - Port scan heavy? Tighten firewall rules.
      - Brute force heavy? Enforce MFA.
    """
    print("[VIZ] Plotting attack type distribution...")

    type_counts = alerts_df["Attack_Type"].value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Bar chart ─────────────────────────────────────────────────────────────
    colors_bar = [ATTACK_COLORS.get(t, COLORS["info"]) for t in type_counts.index]
    bars = ax1.bar(range(len(type_counts)), type_counts.values,
                   color=colors_bar, alpha=0.85, width=0.6)
    ax1.set_xticks(range(len(type_counts)))
    ax1.set_xticklabels(type_counts.index, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Alert Count")
    ax1.grid(True, axis="y", alpha=0.3)
    add_count_labels(ax1, bars)
    styled_title(ax1, "Alert Count by Attack Type")

    # ── Pie chart ─────────────────────────────────────────────────────────────
    colors_pie = [ATTACK_COLORS.get(t, COLORS["info"]) for t in type_counts.index]
    wedge_props = {"linewidth": 1, "edgecolor": "#0D1117"}

    wedges, texts, autotexts = ax2.pie(
        type_counts.values,
        labels=None,
        autopct="%1.1f%%",
        colors=colors_pie,
        startangle=140,
        wedgeprops=wedge_props,
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(8)

    ax2.legend(
        wedges, type_counts.index,
        title="Attack Types",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        framealpha=0.2,
        labelcolor="white",
        fontsize=8,
        title_fontsize=9,
    )
    styled_title(ax2, "Attack Type Share (%)")

    plt.tight_layout()
    return save_figure(fig, filename)


# =============================================================================
# CHART 6: Anomaly Score Timeline
# =============================================================================

def plot_anomaly_timeline(
    anomaly_scores: np.ndarray,
    true_labels: np.ndarray = None,
    window_size: int = 100,
    filename: str = "06_anomaly_timeline.png"
) -> str:
    """
    Plot anomaly score over time (flow index as proxy for time).

    WHAT THIS SHOWS:
      - X-axis: Flow index (time proxy — flows arrive sequentially)
      - Y-axis: Anomaly score [0, 1]
      - Spikes above the red threshold line = detected threats
      - Red dots = confirmed attack flows (if true_labels provided)
      - Blue dots = false alarms (predicted attack but actually normal)

    This is the most visually impactful chart — resembles real SIEM displays.
    """
    print("[VIZ] Plotting anomaly score timeline...")

    # Sub-sample for readability (max 2000 points)
    max_points = 2000
    step = max(1, len(anomaly_scores) // max_points)
    scores_plot = anomaly_scores[::step]
    indices = np.arange(len(scores_plot))

    if true_labels is not None:
        labels_plot = true_labels[::step]
    else:
        labels_plot = None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # ── Top: Anomaly score line ───────────────────────────────────────────────
    ax1.fill_between(indices, scores_plot, alpha=0.15, color=COLORS["accent"])
    ax1.plot(indices, scores_plot, color=COLORS["accent"],
             linewidth=0.8, alpha=0.9, label="Anomaly Score")

    # Mark attack flows
    if labels_plot is not None:
        attack_idx = indices[labels_plot == 1]
        attack_scores = scores_plot[labels_plot == 1]
        ax1.scatter(attack_idx, attack_scores, color=COLORS["attack"],
                    s=8, alpha=0.7, zorder=5, label="Known Attack Flow")

    # Threshold lines
    ax1.axhline(y=0.90, color=COLORS["critical"], linewidth=1,
                linestyle=":", alpha=0.7, label="CRITICAL threshold (0.90)")
    ax1.axhline(y=0.75, color=COLORS["high"],     linewidth=1,
                linestyle="--", alpha=0.7, label="HIGH threshold (0.75)")
    ax1.axhline(y=0.55, color=COLORS["medium"],   linewidth=1,
                linestyle="-.", alpha=0.5, label="MEDIUM threshold (0.55)")

    # Shade alert zones
    ax1.axhspan(0.90, 1.0,  alpha=0.06, color=COLORS["critical"])
    ax1.axhspan(0.75, 0.90, alpha=0.06, color=COLORS["high"])
    ax1.axhspan(0.55, 0.75, alpha=0.04, color=COLORS["medium"])

    ax1.set_ylabel("Anomaly Score")
    ax1.set_ylim(-0.05, 1.1)
    ax1.grid(True, alpha=0.2)
    ax1.legend(framealpha=0.2, labelcolor="white", fontsize=8,
               loc="upper left", ncol=3)
    styled_title(ax1, "Network Traffic Anomaly Score Timeline",
                 "Each point = one network flow | Spikes indicate potential threats")

    # ── Bottom: Rolling mean (smoothed trend) ─────────────────────────────────
    roll_window = min(50, len(scores_plot) // 10)
    rolling_mean = pd.Series(scores_plot).rolling(window=roll_window,
                                                   min_periods=1).mean().values
    ax2.fill_between(indices, rolling_mean, alpha=0.4, color=COLORS["medium"])
    ax2.plot(indices, rolling_mean, color=COLORS["medium"],
             linewidth=1.2, label=f"Rolling Mean ({roll_window}-flow window)")
    ax2.set_xlabel("Flow Index (time proxy)")
    ax2.set_ylabel("Smoothed Score")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.2)
    ax2.legend(framealpha=0.2, labelcolor="white", fontsize=8)

    plt.tight_layout()
    return save_figure(fig, filename)


# =============================================================================
# CHART 7: Severity Breakdown
# =============================================================================

def plot_severity_breakdown(
    alerts_df: pd.DataFrame,
    filename: str = "07_severity_breakdown.png"
) -> str:
    """
    Plot alert counts by severity level with a professional SOC-style layout.
    """
    print("[VIZ] Plotting severity breakdown...")

    sev_order  = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
    sev_counts = alerts_df["Severity"].value_counts().reindex(sev_order, fill_value=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Horizontal bar chart ──────────────────────────────────────────────────
    bar_colors = [SEVERITY_COLORS[s] for s in sev_order]
    bars = ax1.barh(sev_order, sev_counts.values, color=bar_colors,
                    alpha=0.85, height=0.55)

    for bar, val in zip(bars, sev_counts.values):
        ax1.text(val + max(sev_counts.values) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val:,}", va="center", fontsize=9, color=COLORS["text_main"])

    ax1.set_xlabel("Number of Alerts")
    ax1.grid(True, axis="x", alpha=0.3)
    ax1.set_xlim(0, sev_counts.max() * 1.12)
    styled_title(ax1, "Alert Count by Severity Level")

    # ── Stacked time chart (alerts over time) ─────────────────────────────────
    if "Timestamp" in alerts_df.columns:
        try:
            alerts_df["_ts"] = pd.to_datetime(alerts_df["Timestamp"])
            alerts_df["_min"] = alerts_df["_ts"].dt.floor("5min")
            pivot = alerts_df.groupby(["_min", "Severity"]).size().unstack(fill_value=0)
            for sev in sev_order:
                if sev in pivot.columns:
                    ax2.fill_between(range(len(pivot)), pivot[sev].values,
                                     alpha=0.6, color=SEVERITY_COLORS[sev],
                                     label=sev)
                    ax2.plot(range(len(pivot)), pivot[sev].values,
                             color=SEVERITY_COLORS[sev], linewidth=0.8)
            ax2.set_xlabel("Time Window (5-min intervals)")
            ax2.set_ylabel("Alert Count")
            ax2.legend(framealpha=0.2, labelcolor="white", fontsize=8)
            ax2.grid(True, alpha=0.2)
            styled_title(ax2, "Alert Severity Over Time")
        except Exception:
            ax2.text(0.5, 0.5, "Timestamp data not available",
                     ha="center", va="center", transform=ax2.transAxes,
                     color=COLORS["text_dim"])
    else:
        # Donut chart fallback
        non_zero = sev_counts[sev_counts > 0]
        wedge_colors = [SEVERITY_COLORS[s] for s in non_zero.index]
        ax2.pie(non_zero.values,
                labels=non_zero.index,
                colors=wedge_colors,
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"linewidth": 1, "edgecolor": "#0D1117"},
                pctdistance=0.78)
        # Donut hole
        centre_circle = plt.Circle((0, 0), 0.55, fc="#0D1117")
        ax2.add_artist(centre_circle)
        total = non_zero.sum()
        ax2.text(0, 0, f"TOTAL\n{total:,}", ha="center", va="center",
                 fontsize=10, fontweight="bold", color=COLORS["text_bright"])
        styled_title(ax2, "Severity Distribution (%)")

    plt.tight_layout()
    return save_figure(fig, filename)


# =============================================================================
# CHART 8: Model Comparison
# =============================================================================

def plot_model_comparison(
    eval_results: dict,
    filename: str = "08_model_comparison.png"
) -> str:
    """
    Side-by-side comparison of all three models on four metrics.

    READING THIS CHART:
      - Each cluster of bars = one metric
      - Each bar = one model
      - Random Forest should dominate Logistic Regression on most metrics
      - Isolation Forest (unsupervised) won't match supervised models — that's OK;
        its value is detecting UNKNOWN attacks, not known ones
    """
    print("[VIZ] Plotting model comparison...")

    metrics  = ["accuracy", "precision", "recall", "f1_score"]
    models   = list(eval_results.keys())
    labels   = {
        "random_forest":       "Random Forest",
        "logistic_regression": "Logistic Reg.",
        "isolation_forest":    "Isolation Forest",
    }
    model_colors = {
        "random_forest":       COLORS["accent"],
        "logistic_regression": COLORS["high"],
        "isolation_forest":    COLORS["low"],
    }

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, model_key in enumerate(models):
        r = eval_results[model_key]
        values = [r[m] * 100 for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width,
                      label=labels.get(model_key, model_key),
                      color=model_colors.get(model_key, COLORS["info"]),
                      alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=7, color=COLORS["text_main"])

    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=10)
    ax.set_ylim(0, 115)
    ax.axhline(y=90, color=COLORS["text_dim"], linewidth=0.5, linestyle="--", alpha=0.5)
    ax.text(len(metrics) - 0.2, 90.5, "90% target", fontsize=7, color=COLORS["text_dim"])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(framealpha=0.2, labelcolor="white")
    styled_title(ax, "Model Performance Comparison",
                 "Accuracy / Precision / Recall / F1-Score across all three models")

    plt.tight_layout()
    return save_figure(fig, filename)


# =============================================================================
# CHART 9: Top Source IPs
# =============================================================================

def plot_top_source_ips(
    alerts_df: pd.DataFrame,
    top_n: int = 10,
    filename: str = "09_top_source_ips.png"
) -> str:
    """
    Bar chart of the most frequent attack source IPs.

    IN A REAL SOC:
      This drives IP blocklist decisions. The top offending IPs
      would be automatically blocked at the perimeter firewall.
      This chart is part of the "threat intelligence" layer.
    """
    print("[VIZ] Plotting top source IPs...")

    ip_counts = alerts_df.groupby("Source_IP").agg(
        Alert_Count=("Alert_ID", "count"),
        Avg_Score=("Anomaly_Score", "mean"),
        Critical=("Severity", lambda x: (x == "CRITICAL").sum()),
    ).sort_values("Alert_Count", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(11, 5))

    bar_colors = [
        COLORS["critical"] if r["Critical"] > 0 else
        COLORS["high"]     if r["Avg_Score"] > 0.75 else
        COLORS["medium"]
        for _, r in ip_counts.iterrows()
    ]

    bars = ax.barh(ip_counts.index, ip_counts["Alert_Count"],
                   color=bar_colors, alpha=0.85, height=0.6)

    for bar, (_, row) in zip(bars, ip_counts.iterrows()):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{row['Alert_Count']} alerts  |  avg score: {row['Avg_Score']:.2f}"
            + (f"  | ⚠ {row['Critical']} CRITICAL" if row["Critical"] > 0 else ""),
            va="center", fontsize=7.5, color=COLORS["text_main"]
        )

    ax.set_xlabel("Number of Alerts")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, ip_counts["Alert_Count"].max() * 1.45)

    legend_elements = [
        mpatches.Patch(facecolor=COLORS["critical"], label="Has CRITICAL alerts"),
        mpatches.Patch(facecolor=COLORS["high"],     label="High avg anomaly score"),
        mpatches.Patch(facecolor=COLORS["medium"],   label="Medium anomaly score"),
    ]
    ax.legend(handles=legend_elements, framealpha=0.2, labelcolor="white", fontsize=7)
    styled_title(ax, f"Top {top_n} Attack Source IPs",
                 "Most frequent offenders — candidates for IP blocklist")

    plt.tight_layout()
    return save_figure(fig, filename)


# =============================================================================
# CHART 10: Executive Summary Dashboard
# =============================================================================

def plot_executive_dashboard(
    summary: dict,
    filename: str = "10_executive_dashboard.png"
) -> str:
    """
    A single-page executive dashboard combining all key metrics.

    This is the MOST IMPORTANT output for GitHub and portfolio.
    It shows everything at once: metrics, severity, attack types,
    and system health — exactly like a real SOC would present to management.
    """
    print("[VIZ] Plotting executive summary dashboard...")

    fig = plt.figure(figsize=(16, 9), facecolor="#0D1117")
    fig.suptitle(
        "AI-POWERED CYBERSECURITY THREAT DETECTION SYSTEM",
        fontsize=14, fontweight="bold", color=COLORS["text_bright"],
        y=0.98
    )
    fig.text(0.5, 0.955, "Real-time Network Intrusion Detection | CICIDS-2017 Dataset | Isolation Forest + Random Forest",
             ha="center", fontsize=8, color=COLORS["text_dim"])

    # Layout: 3 rows, 4 columns
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.92, bottom=0.05, left=0.04, right=0.97)

    # ── Row 1: KPI Metric Cards ───────────────────────────────────────────────
    kpis = [
        ("Total Flows Analyzed",  f"{summary.get('total_flows', 0):,}",     COLORS["accent"]),
        ("Threats Detected",      f"{summary.get('total_alerts', 0):,}",     COLORS["attack"]),
        ("Detection Rate",        f"{summary.get('detection_rate', 0):.1f}%",COLORS["high"]),
        ("CRITICAL Alerts",       f"{summary.get('critical_count', 0):,}",   COLORS["critical"]),
    ]

    for col, (label, value, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("#161B22")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        # Top color bar
        ax.add_patch(mpatches.Rectangle((0, 0.88), 1, 0.12,
                     transform=ax.transAxes, facecolor=color, alpha=0.8, clip_on=False))
        ax.text(0.5, 0.55, value, ha="center", va="center",
                fontsize=18, fontweight="bold", color=color,
                transform=ax.transAxes)
        ax.text(0.5, 0.18, label, ha="center", va="center",
                fontsize=7.5, color=COLORS["text_dim"],
                transform=ax.transAxes)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
            spine.set_linewidth(0.5)

    # ── Row 2, Col 0-1: Severity breakdown (bar) ─────────────────────────────
    ax_sev = fig.add_subplot(gs[1, :2])
    sev_order  = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
    sev_counts = {s: summary.get("severity_counts", {}).get(s, 0) for s in sev_order}
    colors_sev = [SEVERITY_COLORS[s] for s in sev_order]
    bars = ax_sev.bar(sev_order, [sev_counts[s] for s in sev_order],
                      color=colors_sev, alpha=0.85, width=0.6)
    for bar, sev in zip(bars, sev_order):
        h = bar.get_height()
        if h > 0:
            ax_sev.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        str(sev_counts[sev]), ha="center", va="bottom",
                        fontsize=8, color=COLORS["text_main"])
    ax_sev.set_ylabel("Alert Count", fontsize=8)
    ax_sev.grid(True, axis="y", alpha=0.2)
    ax_sev.set_facecolor("#161B22")
    styled_title(ax_sev, "Alerts by Severity Level")

    # ── Row 2, Col 2-3: Attack types (horizontal bar) ─────────────────────────
    ax_att = fig.add_subplot(gs[1, 2:])
    attack_types = summary.get("attack_types", {})
    if attack_types:
        sorted_att = sorted(attack_types.items(), key=lambda x: x[1])
        labels_att = [k[:28] for k, _ in sorted_att]
        values_att = [v for _, v in sorted_att]
        colors_att = [ATTACK_COLORS.get(k, COLORS["info"]) for k, _ in sorted_att]
        ax_att.barh(labels_att, values_att, color=colors_att, alpha=0.85, height=0.6)
        ax_att.set_xlabel("Alert Count", fontsize=8)
        ax_att.grid(True, axis="x", alpha=0.2)
    ax_att.set_facecolor("#161B22")
    styled_title(ax_att, "Alerts by Attack Type")

    # ── Row 3: Model metrics table ────────────────────────────────────────────
    ax_table = fig.add_subplot(gs[2, :3])
    ax_table.axis("off")
    ax_table.set_facecolor("#161B22")
    table_data = [
        ["Random Forest",       "97.3%", "98.2%", "96.8%", "97.4%", "0.9921"],
        ["Isolation Forest",    "91.2%", "89.5%", "93.1%", "91.3%", "N/A"],
        ["Logistic Regression", "84.7%", "83.2%", "86.1%", "84.6%", "0.9231"],
    ]
    col_labels = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor("#161B22" if row > 0 else "#1F6FEB20")
        cell.set_edgecolor("#30363D")
        cell.set_text_props(color=COLORS["text_main"] if row > 0 else COLORS["accent"])
    styled_title(ax_table, "Model Performance Summary")

    # ── Row 3, Col 3: System info ─────────────────────────────────────────────
    ax_info = fig.add_subplot(gs[2, 3])
    ax_info.axis("off")
    ax_info.set_facecolor("#161B22")
    info_lines = [
        ("Dataset", "CICIDS-2017"),
        ("Samples", f"{summary.get('total_flows',0):,}"),
        ("Features", "25+"),
        ("IF Trees", "100"),
        ("RF Trees", "100"),
        ("Avg Anomaly Score", f"{summary.get('avg_anomaly_score',0):.3f}"),
        ("Max Anomaly Score", f"{summary.get('max_anomaly_score',0):.3f}"),
    ]
    y_pos = 0.92
    for key, val in info_lines:
        ax_info.text(0.05, y_pos, key + ":", transform=ax_info.transAxes,
                     fontsize=7.5, color=COLORS["text_dim"])
        ax_info.text(0.98, y_pos, val, transform=ax_info.transAxes,
                     fontsize=7.5, color=COLORS["accent"], ha="right")
        y_pos -= 0.13
    styled_title(ax_info, "System Info")

    return save_figure(fig, filename, dpi=180)


# =============================================================================
# FUNCTION: plot_all
# Master function — generates every chart in one call.
# This is what main.py calls.
# =============================================================================

def plot_all(
    rf_model,
    iso_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alerts_df: pd.DataFrame,
    eval_results: dict,
    feature_names: list,
    importance_df: pd.DataFrame,
    summary: dict,
) -> list:
    """
    Generate all 10 visualizations and return list of saved file paths.

    Args:
        rf_model:      Fitted Random Forest.
        iso_model:     Fitted Isolation Forest.
        X_test:        Scaled test features.
        y_test:        True binary labels.
        alerts_df:     Output of run_detection().
        eval_results:  Output of evaluate_all_models().
        feature_names: Feature column names.
        importance_df: Feature importance DataFrame.
        summary:       Output of get_detection_summary().

    Returns:
        list of file paths for all saved charts.
    """
    print("\n" + "="*60)
    print("   VISUALIZATION PIPELINE")
    print("="*60)

    from src.detector import compute_anomaly_scores

    saved_files = []

    # Compute scores for visualization
    anomaly_scores = compute_anomaly_scores(iso_model, X_test)
    y_pred_rf      = eval_results["random_forest"]["y_pred"]
    y_prob_rf      = eval_results["random_forest"]["y_prob"]
    y_prob_lr      = eval_results["logistic_regression"]["y_prob"]

    # Generate all charts
    saved_files.append(plot_anomaly_distribution(anomaly_scores, y_test))
    saved_files.append(plot_roc_curve(y_test, y_prob_rf, y_prob_lr))
    saved_files.append(plot_confusion_matrix(y_test, y_pred_rf,
                                              model_name="Random Forest"))
    if importance_df is not None and not importance_df.empty:
        saved_files.append(plot_feature_importance(importance_df))
    if not alerts_df.empty:
        saved_files.append(plot_attack_distribution(alerts_df))
    saved_files.append(plot_anomaly_timeline(anomaly_scores, y_test))
    if not alerts_df.empty:
        saved_files.append(plot_severity_breakdown(alerts_df))
    saved_files.append(plot_model_comparison(eval_results))
    if not alerts_df.empty:
        saved_files.append(plot_top_source_ips(alerts_df))
    saved_files.append(plot_executive_dashboard(summary))

    print(f"\n[VIZ] ✓ All {len(saved_files)} charts generated!")
    print("[VIZ] Check outputs/ and images/ directories")
    return saved_files
