# =============================================================================
# FILE: src/detector.py
# PURPOSE: Real-time threat detection, anomaly scoring, alert generation,
#          severity classification, and threat reporting.
#
# WHAT THIS FILE SIMULATES:
#   In a real SOC (Security Operations Center), this module would sit between
#   the network tap (where raw traffic is captured) and the alerting platform
#   (like PagerDuty, Slack, or a SIEM dashboard). Every incoming network flow
#   gets an anomaly score, a classification, and a severity level.
#
# THREE-LAYER DETECTION LOGIC:
#   Layer 1 — Isolation Forest (Unsupervised)
#             Detects unknown/novel threats by scoring how "isolated" a flow is.
#             Works even when an attack has NEVER been seen before.
#
#   Layer 2 — Random Forest (Supervised)
#             Classifies flow as normal or a specific known attack type.
#             Provides a confidence probability (0.0 to 1.0).
#
#   Layer 3 — Rule-based Severity Engine
#             Combines both scores and applies security domain rules to
#             assign CRITICAL / HIGH / MEDIUM / LOW severity levels.
#
# INDUSTRY RELEVANCE:
#   This hybrid approach (unsupervised + supervised + rules) mirrors how
#   platforms like Darktrace, Vectra AI, and CrowdStrike Falcon work.
# =============================================================================

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# THRESHOLD CONFIGURATION
# These values would normally come from a config.yaml or environment variables.
# Tune these based on your organization's risk tolerance:
#   - Lower thresholds = more alerts = more false positives (noisy)
#   - Higher thresholds = fewer alerts = more missed attacks (risky)
# =============================================================================

THRESHOLDS = {
    "CRITICAL": {"anomaly_score": 0.90, "rf_probability": 0.95},
    "HIGH":     {"anomaly_score": 0.75, "rf_probability": 0.80},
    "MEDIUM":   {"anomaly_score": 0.55, "rf_probability": 0.60},
    "LOW":      {"anomaly_score": 0.35, "rf_probability": 0.40},
}

# Output directory for alert logs
OUTPUTS_DIR = "outputs"

# Known attack signatures for rule-based detection layer
# (These are simplified rules based on domain knowledge)
RULE_SIGNATURES = {
    "DoS_high_pps": {
        "description": "DoS: Flow Packets/s > 10000",
        "feature":     "Flow Packets/s",
        "threshold":   10000,
        "operator":    ">",
    },
    "PortScan_low_duration": {
        "description": "Port Scan: Duration < 100µs + SYN flags",
        "feature":     "Flow Duration",
        "threshold":   100,
        "operator":    "<",
    },
    "BruteForce_high_bps": {
        "description": "Brute Force: High repetitive byte rate",
        "feature":     "Flow Bytes/s",
        "threshold":   50000,
        "operator":    ">",
    },
}


# =============================================================================
# FUNCTION: compute_anomaly_scores
# Converts raw Isolation Forest output to a normalized [0, 1] anomaly score.
# =============================================================================

def compute_anomaly_scores(iso_model, X: np.ndarray) -> np.ndarray:
    """
    Compute normalized anomaly scores using Isolation Forest.

    RAW OUTPUT EXPLAINED:
      iso_model.score_samples(X) returns negative average path lengths.
        - More negative = shorter path = more anomalous
        - Less negative = longer path  = more normal

    NORMALIZATION:
      We scale these to [0, 1] where:
        - Score near 1.0 = highly anomalous (very likely an attack)
        - Score near 0.0 = very normal (likely benign traffic)

    Args:
        iso_model:     Fitted IsolationForest.
        X (np.ndarray): Scaled feature matrix.

    Returns:
        np.ndarray of shape (n_samples,) with scores in [0, 1].
    """
    # Get raw scores — more negative = more anomalous
    raw_scores = iso_model.score_samples(X)

    # Normalize to [0, 1] range using min-max scaling
    score_min = raw_scores.min()
    score_max = raw_scores.max()

    if score_max == score_min:
        # Edge case: all scores are identical
        return np.zeros(len(raw_scores))

    # Invert: so high score = high anomaly
    normalized = 1.0 - (raw_scores - score_min) / (score_max - score_min)
    return normalized


# =============================================================================
# FUNCTION: compute_rf_probabilities
# Gets Random Forest attack probability for each flow.
# =============================================================================

def compute_rf_probabilities(rf_model, X: np.ndarray) -> tuple:
    """
    Get Random Forest predictions and attack probabilities.

    Args:
        rf_model:  Fitted RandomForestClassifier.
        X:         Scaled feature matrix.

    Returns:
        tuple: (predictions array, attack_probabilities array)
               predictions:         0 (normal) or 1 (attack)
               attack_probabilities: float in [0, 1] per sample
    """
    # predict() gives the class label (0 or 1)
    predictions = rf_model.predict(X)

    # predict_proba() gives [prob_class0, prob_class1] per sample
    probabilities = rf_model.predict_proba(X)

    # We want the probability of class 1 (Attack)
    attack_probs = probabilities[:, 1]

    return predictions, attack_probs


# =============================================================================
# FUNCTION: classify_severity
# Combines anomaly score + RF probability into a severity level.
# =============================================================================

def classify_severity(
    anomaly_score: float,
    rf_probability: float,
    rf_prediction: int
) -> str:
    """
    Classify threat severity using a combined scoring engine.

    SEVERITY LOGIC:
      We use a conservative approach: we take the MAXIMUM severity
      implied by either the anomaly score OR the RF probability.
      This reduces the chance of missing a severe attack.

      CRITICAL: Score ≥ 0.90 or RF prob ≥ 0.95 (very high confidence)
      HIGH:     Score ≥ 0.75 or RF prob ≥ 0.80
      MEDIUM:   Score ≥ 0.55 or RF prob ≥ 0.60
      LOW:      Score ≥ 0.35 or RF prob ≥ 0.40
      INFO:     Below all thresholds but RF predicted attack

    Args:
        anomaly_score  (float): Normalized Isolation Forest score [0,1].
        rf_probability (float): RF attack probability [0,1].
        rf_prediction  (int):   RF class prediction (0=normal, 1=attack).

    Returns:
        str: "CRITICAL", "HIGH", "MEDIUM", "LOW", or "INFO"
    """
    # Check from highest to lowest severity
    if (anomaly_score >= THRESHOLDS["CRITICAL"]["anomaly_score"] or
            rf_probability >= THRESHOLDS["CRITICAL"]["rf_probability"]):
        return "CRITICAL"

    if (anomaly_score >= THRESHOLDS["HIGH"]["anomaly_score"] or
            rf_probability >= THRESHOLDS["HIGH"]["rf_probability"]):
        return "HIGH"

    if (anomaly_score >= THRESHOLDS["MEDIUM"]["anomaly_score"] or
            rf_probability >= THRESHOLDS["MEDIUM"]["rf_probability"]):
        return "MEDIUM"

    if (anomaly_score >= THRESHOLDS["LOW"]["anomaly_score"] or
            rf_probability >= THRESHOLDS["LOW"]["rf_probability"]):
        return "LOW"

    # Both models suggest it's suspicious but below thresholds
    if rf_prediction == 1:
        return "INFO"

    return "INFO"


# =============================================================================
# FUNCTION: infer_attack_type
# Uses RF probability + anomaly score pattern to guess attack type.
# This is simplified heuristics — real systems would use multi-class RF.
# =============================================================================

def infer_attack_type(
    anomaly_score: float,
    rf_probability: float,
    sample_features: np.ndarray,
    feature_names: list
) -> str:
    """
    Heuristically infer attack type from feature patterns.

    In a production system, you'd use a multi-class Random Forest directly.
    This function demonstrates rule-based domain knowledge.

    Args:
        anomaly_score:   Normalized IF score.
        rf_probability:  RF attack probability.
        sample_features: The feature vector for this single flow.
        feature_names:   Names of the features.

    Returns:
        str: Inferred attack type label.
    """
    if rf_probability < 0.40 and anomaly_score < 0.40:
        return "Benign"

    # Build a small lookup from feature name → value
    feat_dict = {}
    if feature_names is not None and len(feature_names) == len(sample_features):
        for name, val in zip(feature_names, sample_features):
            feat_dict[name.strip().lower()] = val

    # Heuristic rules based on domain knowledge about CICFlowMeter features
    pps = feat_dict.get("flow packets/s", 0)
    bps = feat_dict.get("flow bytes/s", 0)
    dur = feat_dict.get("flow duration", 9999999)
    syn = feat_dict.get("syn flag count", 0)
    rst = feat_dict.get("rst flag count", 0)
    psh = feat_dict.get("fwd psh flags", 0)

    if pps > 5000 or bps > 500000:
        return "DoS / DDoS Attack"
    if dur < 500 and (syn > 2 or rst > 2):
        return "Port Scan"
    if psh > 0 and bps > 20000:
        return "Brute Force"
    if 1000 < bps < 50000 and dur > 10000:
        return "Web Attack"
    if 0 < pps < 50 and dur > 30000:
        return "Botnet / C2 Activity"
    if anomaly_score > 0.75:
        return "Unknown / Novel Attack"

    return "Suspicious Activity"


# =============================================================================
# FUNCTION: generate_fake_ip
# Simulates source/destination IP addresses for alert realism.
# =============================================================================

def generate_fake_ip(rng: np.random.Generator, is_external: bool = True) -> str:
    """
    Generate a fake but realistic-looking IP address.

    In real detection, this would come from the pcap metadata.
    We simulate it here because our feature-only dataset doesn't contain IPs.

    Args:
        rng:         NumPy random generator for reproducibility.
        is_external: If True, generates a routable public IP.
                     If False, generates a private subnet IP.
    """
    if is_external:
        # Exclude reserved ranges: 10.x, 172.16-31.x, 192.168.x, 127.x
        first = rng.choice([
            11, 13, 14, 15, 16, 17, 18, 20, 23, 24,
            34, 35, 52, 54, 66, 67, 74, 99, 104, 108,
            198, 199, 203, 208, 209, 216, 218, 220
        ])
        return f"{first}.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,254)}"
    else:
        prefix = rng.choice(["10.0", "10.10", "172.16", "192.168.1", "192.168.2"])
        return f"{prefix}.{rng.integers(1,254)}.{rng.integers(1,254)}"


# =============================================================================
# FUNCTION: run_detection
# MAIN DETECTION ENGINE — processes all test flows and generates alerts.
# =============================================================================

def run_detection(
    iso_model,
    rf_model,
    X: np.ndarray,
    feature_names: list = None,
    threshold_override: float = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run the full three-layer threat detection pipeline on a dataset.

    PIPELINE:
      1. Compute Isolation Forest anomaly score for each flow
      2. Compute Random Forest prediction + attack probability
      3. Combine scores to assign severity
      4. Infer probable attack type
      5. Build alert records with simulated metadata

    Args:
        iso_model:          Fitted IsolationForest.
        rf_model:           Fitted RandomForestClassifier.
        X:                  Scaled feature matrix (all test flows).
        feature_names:      Column names for the features (optional).
        threshold_override: Override default LOW threshold for alert filtering.
        random_state:       Seed for reproducible simulated metadata.

    Returns:
        pd.DataFrame with one row per detected threat, sorted by severity.
    """
    print("\n" + "="*60)
    print("   THREAT DETECTION ENGINE")
    print("="*60)
    print(f"[DETECT] Processing {len(X):,} network flows...")

    rng = np.random.default_rng(random_state)
    start_time = time.time()

    # ── Layer 1: Isolation Forest anomaly scores ──────────────────────────────
    print("[DETECT] Layer 1: Computing anomaly scores (Isolation Forest)...")
    anomaly_scores = compute_anomaly_scores(iso_model, X)

    # ── Layer 2: Random Forest classification ────────────────────────────────
    print("[DETECT] Layer 2: Classifying flows (Random Forest)...")
    rf_predictions, rf_probabilities = compute_rf_probabilities(rf_model, X)

    # ── Layer 3: Build alert records ──────────────────────────────────────────
    print("[DETECT] Layer 3: Applying severity engine and building alerts...")

    alerts = []
    alert_threshold = threshold_override or THRESHOLDS["LOW"]["anomaly_score"]

    # Simulate timestamps starting from "1 hour ago"
    base_time = datetime.now() - timedelta(hours=1)

    for i in range(len(X)):
        score  = anomaly_scores[i]
        prob   = rf_probabilities[i]
        pred   = int(rf_predictions[i])

        # Only flag flows that exceed the minimum threshold or are predicted as attacks
        if score >= alert_threshold or pred == 1:
            severity = classify_severity(score, prob, pred)

            # Skip INFO-only if they're clearly below radar
            if severity == "INFO" and score < 0.30 and prob < 0.30:
                continue

            attack_type = infer_attack_type(
                score, prob, X[i], feature_names
            )

            # Simulate source/destination metadata
            is_ext_src = rng.random() > 0.4   # 60% attacks from external IPs
            src_ip  = generate_fake_ip(rng, is_external=is_ext_src)
            dst_ip  = generate_fake_ip(rng, is_external=False)
            dst_port = rng.choice([21, 22, 23, 80, 443, 445, 3389, 8080, 8443])

            # Simulate timestamp (spread over 1 hour)
            offset_seconds = rng.integers(0, 3600)
            timestamp = (base_time + timedelta(seconds=int(offset_seconds))).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            alerts.append({
                "Timestamp":      timestamp,
                "Alert_ID":       f"ALERT-{len(alerts)+1:05d}",
                "Severity":       severity,
                "Attack_Type":    attack_type,
                "Anomaly_Score":  round(float(score), 4),
                "RF_Probability": round(float(prob), 4),
                "RF_Prediction":  "Attack" if pred == 1 else "Normal",
                "Source_IP":      src_ip,
                "Dest_IP":        dst_ip,
                "Dest_Port":      int(dst_port),
                "Flow_Index":     i,
            })

    elapsed = time.time() - start_time

    # ── Build DataFrame and sort by severity ──────────────────────────────────
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    alerts_df = pd.DataFrame(alerts)

    if not alerts_df.empty:
        alerts_df["_sev_rank"] = alerts_df["Severity"].map(severity_order)
        alerts_df = alerts_df.sort_values("_sev_rank").drop("_sev_rank", axis=1)
        alerts_df = alerts_df.reset_index(drop=True)

    # ── Print detection summary ───────────────────────────────────────────────
    print(f"\n[DETECT] ✓ Detection completed in {elapsed:.2f}s")
    print(f"[DETECT] Total flows processed:  {len(X):,}")
    print(f"[DETECT] Total alerts generated: {len(alerts_df):,}")

    if not alerts_df.empty:
        print(f"\n[DETECT] Alerts by severity:")
        sev_counts = alerts_df["Severity"].value_counts()
        sev_colors = {"CRITICAL": "!!!", "HIGH": "!! ", "MEDIUM": "!  ", "LOW": "   ", "INFO": "   "}
        for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
            if sev in sev_counts:
                bar = "█" * min(sev_counts[sev] // 5, 40)
                print(f"  {sev_colors[sev]} {sev:<8} {sev_counts[sev]:>6,}  {bar}")

        print(f"\n[DETECT] Alerts by attack type:")
        for atype, cnt in alerts_df["Attack_Type"].value_counts().items():
            print(f"  {atype:<35} → {cnt:>5,}")

    return alerts_df


# =============================================================================
# FUNCTION: save_alerts
# Exports alert log to CSV for downstream analysis / SIEM ingestion.
# =============================================================================

def save_alerts(alerts_df: pd.DataFrame, filename: str = "alerts.csv") -> str:
    """
    Save the alert DataFrame to a CSV file.

    In production, this would write to:
      - A SIEM platform (Splunk, IBM QRadar) via API
      - A database (PostgreSQL, Elasticsearch)
      - A message queue (Kafka) for real-time streaming
      - A ticketing system (JIRA, ServiceNow)

    For our demo, we write to outputs/alerts.csv.

    Args:
        alerts_df (pd.DataFrame): Alerts from run_detection().
        filename (str):           Output filename.

    Returns:
        str: Full path to saved file.
    """
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUTS_DIR, filename)
    alerts_df.to_csv(filepath, index=False)
    print(f"\n[SAVE] Alerts exported → {filepath}")
    print(f"       {len(alerts_df):,} records | {os.path.getsize(filepath)/1024:.1f} KB")
    return filepath


# =============================================================================
# FUNCTION: get_detection_summary
# Returns a structured summary report for the dashboard.
# =============================================================================

def get_detection_summary(alerts_df: pd.DataFrame, total_flows: int) -> dict:
    """
    Build a summary statistics dictionary from the alert DataFrame.

    Used by the visualizer to generate the executive summary page.

    Args:
        alerts_df (pd.DataFrame): Output of run_detection().
        total_flows (int):        Total number of flows processed.

    Returns:
        dict with summary statistics.
    """
    if alerts_df.empty:
        return {
            "total_flows":      total_flows,
            "total_alerts":     0,
            "detection_rate":   0.0,
            "severity_counts":  {},
            "attack_types":     {},
            "top_source_ips":   {},
            "top_dest_ports":   {},
            "avg_anomaly_score": 0.0,
            "max_anomaly_score": 0.0,
        }

    sev_counts = alerts_df["Severity"].value_counts().to_dict()
    attack_types = alerts_df["Attack_Type"].value_counts().to_dict()
    top_src_ips = alerts_df["Source_IP"].value_counts().head(10).to_dict()
    top_ports = alerts_df["Dest_Port"].value_counts().head(10).to_dict()

    return {
        "total_flows":       total_flows,
        "total_alerts":      len(alerts_df),
        "detection_rate":    round(len(alerts_df) / total_flows * 100, 2),
        "critical_count":    sev_counts.get("CRITICAL", 0),
        "high_count":        sev_counts.get("HIGH", 0),
        "medium_count":      sev_counts.get("MEDIUM", 0),
        "low_count":         sev_counts.get("LOW", 0),
        "severity_counts":   sev_counts,
        "attack_types":      attack_types,
        "top_source_ips":    top_src_ips,
        "top_dest_ports":    top_ports,
        "avg_anomaly_score": round(alerts_df["Anomaly_Score"].mean(), 4),
        "max_anomaly_score": round(alerts_df["Anomaly_Score"].max(), 4),
        "avg_rf_probability":round(alerts_df["RF_Probability"].mean(), 4),
    }


# =============================================================================
# FUNCTION: simulate_realtime_stream
# Simulates real-time detection on a stream of incoming flows.
# Great for demos — shows detection happening "live".
# =============================================================================

def simulate_realtime_stream(
    iso_model,
    rf_model,
    X: np.ndarray,
    feature_names: list = None,
    n_flows: int = 20,
    delay_seconds: float = 0.3,
    random_state: int = 42
):
    """
    Simulate real-time threat detection by processing flows one at a time.

    This function prints each detection result as it processes, mimicking
    how a live SIEM would display incoming alerts.

    Args:
        iso_model:      Fitted IsolationForest.
        rf_model:       Fitted RandomForestClassifier.
        X:              Full scaled feature matrix.
        feature_names:  Feature names for attack type inference.
        n_flows:        How many flows to simulate.
        delay_seconds:  Pause between each flow (for visual effect).
        random_state:   Seed for reproducible simulation.
    """
    rng = np.random.default_rng(random_state)

    # Pick random samples to simulate
    indices = rng.choice(len(X), size=min(n_flows, len(X)), replace=False)

    print("\n" + "="*65)
    print("  REAL-TIME STREAM SIMULATION")
    print("="*65)
    print(f"  {'TIME':<10} {'SEVERITY':<10} {'TYPE':<30} {'SCORE':<7} {'PROB':<7}")
    print("-"*65)

    for idx in indices:
        flow = X[idx:idx+1]

        # Compute scores
        score = float(compute_anomaly_scores(iso_model, flow)[0])
        pred, probs = compute_rf_probabilities(rf_model, flow)
        prob = float(probs[0])

        # Classify
        severity = classify_severity(score, prob, int(pred[0]))
        atype = infer_attack_type(score, prob, X[idx], feature_names)

        # Format severity label with visual indicator
        sev_markers = {
            "CRITICAL": "[!!!]",
            "HIGH":     "[!! ]",
            "MEDIUM":   "[!  ]",
            "LOW":      "[.  ]",
            "INFO":     "[   ]",
        }
        ts = datetime.now().strftime("%H:%M:%S")
        marker = sev_markers.get(severity, "[?  ]")

        # Only print if it's an alert-worthy flow
        if severity not in ("INFO",) or pred[0] == 1:
            print(f"  {ts:<10} {marker} {severity:<6} {atype:<30} {score:.3f}   {prob:.3f}")

        time.sleep(delay_seconds)

    print("-"*65)
    print("  Stream simulation complete.\n")
