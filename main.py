# =============================================================================
# FILE: main.py
# PURPOSE: Central orchestrator — runs the complete AI Cybersecurity
#          Threat Detection pipeline from start to finish.
#
# HOW TO RUN:
#   python main.py                     ← uses synthetic demo dataset
#   python main.py --real              ← uses real CICIDS-2017 dataset
#   python main.py --data path/to.csv  ← uses a specific CSV file
#   python main.py --simulate          ← runs real-time stream simulation
#   python main.py --rows 50000        ← limit rows for fast testing
#
# PIPELINE STAGES:
#   Stage 1 — Preprocessing    (preprocess.py)
#   Stage 2 — Model Training   (model.py)
#   Stage 3 — Model Evaluation (model.py)
#   Stage 4 — Threat Detection (detector.py)
#   Stage 5 — Visualization    (visualizer.py)
#   Stage 6 — Report + Summary
#
# PROJECT: AI-Powered Cybersecurity Threat Detection System
# DATASET: CICIDS-2017 (Canadian Institute for Cybersecurity)
# MODELS:  Isolation Forest (unsupervised) + Random Forest (supervised)
#          + Logistic Regression (baseline)
# =============================================================================

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

# ── Import our custom modules ─────────────────────────────────────────────────
# These are the files we built in src/
from src.preprocessing import (
    full_preprocessing_pipeline,
)
from src.model import (
    train_all_models,
    evaluate_all_models,
    get_feature_importance,
    compare_models,
)
from src.detector import (
    run_detection,
    save_alerts,
    get_detection_summary,
    simulate_realtime_stream,
    compute_anomaly_scores,
)
from src.visualizer import plot_all


# =============================================================================
# CONFIGURATION
# Centralize all project settings here.
# In production, these would come from a config.yaml or environment variables.
# =============================================================================

CONFIG = {
    # ── Dataset settings ──────────────────────────────────────────────────────
    "data_path":        "data/raw/",           # Folder with CICIDS-2017 CSVs
    "demo_samples":     50000,                  # Samples when using demo mode
    "test_size":        0.20,                   # 20% of data for testing

    # ── Model settings ────────────────────────────────────────────────────────
    "if_contamination": 0.10,                   # Expected attack rate for IsolationForest
    "rf_n_estimators":  100,                    # Number of trees in Random Forest
    "rf_max_depth":     None,                   # Max tree depth (None = unlimited)
    "random_state":     42,                     # For reproducibility

    # ── Detection settings ────────────────────────────────────────────────────
    "alert_threshold":  0.35,                   # Minimum anomaly score to generate alert

    # ── Output settings ───────────────────────────────────────────────────────
    "output_dir":       "outputs/",
    "models_dir":       "models/",
    "images_dir":       "images/",
    "alert_filename":   "alerts.csv",
    "report_filename":  "detection_report.txt",
}


# =============================================================================
# BANNER
# Prints a professional startup banner. Makes demos look impressive.
# =============================================================================

def print_banner():
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     AI-POWERED CYBERSECURITY THREAT DETECTION SYSTEM         ║
║                                                              ║
║     Models   :  Isolation Forest + Random Forest             ║
║     Dataset  :  CICIDS-2017 Network Traffic                  ║
║     Purpose  :  Detect intrusions, DoS, Brute Force,         ║
║                 Port Scans, Web Attacks, Botnets             ║
║                                                              ║
║     [ Simulating real-world SOC environment ]                ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


# =============================================================================
# STAGE PRINTER
# Makes stage transitions clear in terminal output.
# =============================================================================

def print_stage(number: int, title: str):
    """Print a clear stage separator to terminal."""
    print(f"\n{'━'*62}")
    print(f"  STAGE {number}  ─  {title.upper()}")
    print(f"{'━'*62}")


# =============================================================================
# FUNCTION: setup_directories
# Creates all required project directories.
# =============================================================================

def setup_directories():
    """Ensure all required directories exist before the pipeline runs."""
    dirs = [
        CONFIG["output_dir"],
        CONFIG["models_dir"],
        CONFIG["images_dir"],
        "data/raw/",
        "data/processed/",
        "notebooks/",
        "docs/",
        "src/",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # Create __init__.py so src/ is a proper Python package
    init_path = "src/__init__.py"
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("# src package\n")

    print("[SETUP] All directories ready.")


# =============================================================================
# FUNCTION: save_report
# Saves a text summary of the full detection run.
# =============================================================================

def save_report(summary: dict, eval_results: dict) -> str:
    """
    Write a plain-text detection report to outputs/.

    In production, this would be formatted as a PDF or sent via email
    to the security team or CISO.

    Args:
        summary:      From get_detection_summary().
        eval_results: From evaluate_all_models().

    Returns:
        str: Path to the saved report.
    """
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    filepath = os.path.join(CONFIG["output_dir"], CONFIG["report_filename"])

    lines = [
        "=" * 60,
        "  AI CYBERSECURITY THREAT DETECTION — RUN REPORT",
        "=" * 60,
        "",
        f"  Total Flows Processed  : {summary.get('total_flows', 0):,}",
        f"  Total Alerts Generated : {summary.get('total_alerts', 0):,}",
        f"  Detection Rate         : {summary.get('detection_rate', 0):.2f}%",
        f"  Average Anomaly Score  : {summary.get('avg_anomaly_score', 0):.4f}",
        f"  Max Anomaly Score      : {summary.get('max_anomaly_score', 0):.4f}",
        "",
        "  ALERTS BY SEVERITY:",
    ]

    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        count = summary.get("severity_counts", {}).get(sev, 0)
        if count > 0:
            lines.append(f"    {sev:<10}: {count:,}")

    lines += [
        "",
        "  TOP ATTACK TYPES:",
    ]
    for atype, cnt in list(summary.get("attack_types", {}).items())[:6]:
        lines.append(f"    {atype:<35}: {cnt:,}")

    lines += [
        "",
        "=" * 60,
        "  MODEL PERFORMANCE:",
        "=" * 60,
    ]
    for model_key, result in eval_results.items():
        lines += [
            f"  {result['model_name']}:",
            f"    Accuracy  : {result['accuracy']*100:.2f}%",
            f"    Precision : {result['precision']*100:.2f}%",
            f"    Recall    : {result['recall']*100:.2f}%",
            f"    F1-Score  : {result['f1_score']*100:.2f}%",
            "",
        ]

    lines += [
        "=" * 60,
        "  All output files saved to: outputs/",
        "  All chart images saved to: images/",
        "=" * 60,
    ]

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    print(f"\n[REPORT] Text report saved → {filepath}")
    return filepath


# =============================================================================
# FUNCTION: parse_args
# Command-line argument parsing for flexible usage.
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Cybersecurity Threat Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Run with synthetic demo data (recommended for first run)
  python main.py --real                 # Use real CICIDS-2017 data from data/raw/
  python main.py --data my_file.csv     # Use a specific CSV file
  python main.py --rows 30000           # Limit to 30,000 rows for speed
  python main.py --simulate             # Run real-time stream simulation after training
  python main.py --no-viz               # Skip visualization (faster run)
        """
    )

    parser.add_argument(
        "--real", action="store_true",
        help="Use real CICIDS-2017 dataset from data/raw/ instead of demo data"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to a specific CSV file or folder"
    )
    parser.add_argument(
        "--rows", type=int, default=None,
        help="Maximum rows to load (useful for quick testing on laptops)"
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run real-time threat stream simulation after training"
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip visualization generation (faster pipeline for CI/CD)"
    )
    parser.add_argument(
        "--demo-samples", type=int, default=50000,
        help="Number of synthetic samples to generate in demo mode (default: 50000)"
    )

    return parser.parse_args()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Execute the complete AI Cybersecurity Threat Detection pipeline.

    Pipeline stages:
      0 → Setup
      1 → Preprocessing (load, clean, encode, feature select, scale)
      2 → Model Training (Isolation Forest + Random Forest + LR)
      3 → Model Evaluation (accuracy, precision, recall, F1, AUC-ROC)
      4 → Threat Detection (anomaly scoring, severity, alert generation)
      5 → Visualization (10 charts saved to outputs/ and images/)
      6 → Final Report
    """
    # ── Parse command-line arguments ──────────────────────────────────────────
    args = parse_args()

    # ── Print banner ──────────────────────────────────────────────────────────
    print_banner()
    total_start = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 0: SETUP
    # ══════════════════════════════════════════════════════════════════════════
    print_stage(0, "Environment Setup")
    setup_directories()

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1: PREPROCESSING
    # ══════════════════════════════════════════════════════════════════════════
    print_stage(1, "Data Loading & Preprocessing")

    # Decide whether to use real data or demo data
    use_demo  = not args.real and args.data is None
    data_path = args.data or CONFIG["data_path"]

    if use_demo:
        print("[INFO] Running in DEMO MODE with synthetic data.")
        print("[INFO] To use real data: python main.py --real")
        print("[INFO] Download CICIDS-2017 from: https://www.unb.ca/cic/datasets/ids-2017.html")
    else:
        print(f"[INFO] Using real data from: {data_path}")

    # Run the full preprocessing pipeline
    data = full_preprocessing_pipeline(
        data_path=data_path,
        max_rows=args.rows,
        test_size=CONFIG["test_size"],
        use_demo=use_demo,
        demo_samples=args.demo_samples,
    )

    # Unpack preprocessed data components
    X_train        = data["X_train"]
    X_test         = data["X_test"]
    y_train_binary = data["y_train_binary"]
    y_test_binary  = data["y_test_binary"]
    y_train_multi  = data["y_train_multi"]
    y_test_multi   = data["y_test_multi"]
    scaler         = data["scaler"]
    feature_names  = data["feature_names"]
    le_binary      = data["le_binary"]
    le_multi       = data["le_multi"]
    df_processed   = data["df_processed"]

    # Save processed data to CSV for inspection
    proc_path = "data/processed/processed_data_sample.csv"
    os.makedirs("data/processed", exist_ok=True)
    df_processed.head(1000).to_csv(proc_path, index=False)
    print(f"\n[SAVE] Processed data sample saved → {proc_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2: MODEL TRAINING
    # ══════════════════════════════════════════════════════════════════════════
    print_stage(2, "Model Training")

    models = train_all_models(
        X_train=X_train,
        y_train_binary=y_train_binary,
        feature_names=feature_names,
    )

    # Extract individual models for later use
    iso_model = models["isolation_forest"]
    rf_model  = models["random_forest"]
    lr_model  = models["logistic_regression"]

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3: MODEL EVALUATION
    # ══════════════════════════════════════════════════════════════════════════
    print_stage(3, "Model Evaluation")

    eval_results = evaluate_all_models(
        models=models,
        X_test=X_test,
        y_test_binary=y_test_binary,
        le_multi=le_multi,
    )

    # Get feature importances
    importance_df = get_feature_importance(rf_model, feature_names, top_n=20)

    # Save importance table to CSV
    imp_path = os.path.join(CONFIG["output_dir"], "feature_importances.csv")
    importance_df.to_csv(imp_path, index=False)
    print(f"[SAVE] Feature importances → {imp_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 4: THREAT DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    print_stage(4, "Threat Detection & Alert Generation")

    alerts_df = run_detection(
        iso_model=iso_model,
        rf_model=rf_model,
        X=X_test,
        feature_names=feature_names,
        threshold_override=CONFIG["alert_threshold"],
        random_state=CONFIG["random_state"],
    )

    # Save alerts to CSV
    if not alerts_df.empty:
        save_alerts(alerts_df, filename=CONFIG["alert_filename"])
    else:
        print("[WARN] No alerts generated. Try lowering the alert threshold.")

    # Build detection summary (used for report + dashboard chart)
    summary = get_detection_summary(alerts_df, total_flows=len(X_test))

    # ── Optional: Real-time stream simulation ─────────────────────────────────
    if args.simulate:
        print_stage("4b", "Real-Time Stream Simulation")
        simulate_realtime_stream(
            iso_model=iso_model,
            rf_model=rf_model,
            X=X_test,
            feature_names=feature_names,
            n_flows=25,
            delay_seconds=0.25,
            random_state=CONFIG["random_state"],
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 5: VISUALIZATION
    # ══════════════════════════════════════════════════════════════════════════
    if not args.no_viz:
        print_stage(5, "Visualization (10 Charts)")
        saved_charts = plot_all(
            rf_model=rf_model,
            iso_model=iso_model,
            X_test=X_test,
            y_test=y_test_binary,
            alerts_df=alerts_df,
            eval_results=eval_results,
            feature_names=feature_names,
            importance_df=importance_df,
            summary=summary,
        )
    else:
        print("[INFO] Visualization skipped (--no-viz flag set)")
        saved_charts = []

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 6: FINAL REPORT
    # ══════════════════════════════════════════════════════════════════════════
    print_stage(6, "Final Report")
    save_report(summary, eval_results)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{'═'*62}")
    print(f"  PIPELINE COMPLETE  —  Total time: {total_elapsed:.1f}s")
    print(f"{'═'*62}")
    print(f"\n  Flows processed   : {summary.get('total_flows', 0):,}")
    print(f"  Threats detected  : {summary.get('total_alerts', 0):,}")
    print(f"  Detection rate    : {summary.get('detection_rate', 0):.2f}%")
    print(f"  CRITICAL alerts   : {summary.get('critical_count', 0):,}")
    print(f"  Charts generated  : {len(saved_charts)}")
    print(f"\n  Output files:")
    print(f"    outputs/alerts.csv          ← All detected threats")
    print(f"    outputs/detection_report.txt ← Text summary")
    print(f"    outputs/*.png               ← All charts")
    print(f"    images/*.png                ← Same charts (for README)")
    print(f"    models/*.pkl                ← Saved trained models")
    print(f"\n  GitHub-ready outputs are in: images/")
    print(f"  Add to README.md with: ![Chart](images/10_executive_dashboard.png)")
    print(f"\n  {'═'*60}")
    print(f"  Project by: [YOUR NAME]")
    print(f"  Dataset : CICIDS-2017 (Canadian Inst. for Cybersecurity)")
    print(f"  GitHub  : github.com/[YOUR_USERNAME]/AI-Cybersecurity")
    print(f"  {'═'*60}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
