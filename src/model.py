# =============================================================================
# FILE: src/model.py
# PURPOSE: Train, evaluate, save, and load ML models for threat detection.
#
# MODELS USED:
#   1. Isolation Forest  — Unsupervised anomaly detector. Learns what
#      "normal" traffic looks like and flags deviations. Does NOT need
#      labeled attack data to detect new/unknown threats. Ideal for
#      zero-day attack detection.
#
#   2. Random Forest Classifier — Supervised multi-class classifier.
#      Trained on labeled samples to recognize specific attack types.
#      Produces both a class prediction and a confidence probability.
#
#   3. Logistic Regression — Simple baseline. Used to compare against
#      the more powerful models and demonstrate improvement.
#
# INDUSTRY CONTEXT:
#   Real SIEM systems (Splunk, IBM QRadar, Microsoft Sentinel) combine
#   unsupervised anomaly detection with supervised classifiers — exactly
#   this hybrid approach. Isolation Forest handles unknown threats;
#   Random Forest identifies known attack patterns with high confidence.
# =============================================================================

import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings("ignore")

# Path where trained models are saved
MODELS_DIR = "models"


# =============================================================================
# HELPER: ensure_models_dir
# =============================================================================

def ensure_models_dir():
    """Create the models/ directory if it doesn't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)


# =============================================================================
# FUNCTION: train_isolation_forest
# Unsupervised anomaly detector — no labels needed during training.
# =============================================================================

def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.1,
    n_estimators: int = 100,
    random_state: int = 42
) -> IsolationForest:
    """
    Train an Isolation Forest model for unsupervised anomaly detection.

    HOW ISOLATION FOREST WORKS (simple explanation):
      - Randomly pick a feature and a split point
      - Anomalies (attacks) are "isolated" in fewer splits because they
        are rare and statistically different from normal data
      - Score = average path length to isolate a point
        → Short path = anomaly (attack)
        → Long path  = normal traffic

    Args:
        X_train (np.ndarray):  Scaled training feature matrix.
        contamination (float): Expected fraction of anomalies (0.1 = 10%).
                               For CICIDS-2017, ~20% are attacks, but we
                               use 0.1 conservatively to reduce false positives.
        n_estimators (int):    Number of isolation trees (more = more stable).
        random_state (int):    Seed for reproducibility.

    Returns:
        Fitted IsolationForest model.
    """
    print("\n[MODEL] Training Isolation Forest (unsupervised)...")
    print(f"        Estimators: {n_estimators} | Contamination: {contamination}")

    start = time.time()
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",       # uses min(256, n_samples) sub-samples per tree
        bootstrap=False,          # no replacement sampling — better anomaly isolation
        n_jobs=-1,                # use all CPU cores
        random_state=random_state,
        verbose=0,
    )
    model.fit(X_train)
    elapsed = time.time() - start

    print(f"  [+] Training completed in {elapsed:.2f}s")
    print(f"  [+] Offset (contamination threshold): {model.offset_:.4f}")
    return model


# =============================================================================
# FUNCTION: train_random_forest
# Supervised classifier for known attack type recognition.
# =============================================================================

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = None,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on labeled network traffic data.

    HOW RANDOM FOREST WORKS (simple explanation):
      - Builds many decision trees, each trained on a random subset of
        the data and features
      - For prediction: each tree votes, majority vote wins
      - "Random" + "Forest" = diversity prevents overfitting
      - Returns predict_proba() — a confidence score per class

    WHY RANDOM FOREST FOR CYBERSECURITY:
      - Handles class imbalance well (class_weight='balanced')
      - Feature importance tells us WHICH network features matter most
      - Resistant to noisy/corrupted network flows
      - Fast inference — critical for real-time detection

    Args:
        X_train (np.ndarray): Scaled training features.
        y_train (np.ndarray): Binary or multi-class labels.
        n_estimators (int):   Number of trees.
        max_depth (int):      Max tree depth. None = unlimited (may overfit).
        random_state (int):   Seed for reproducibility.

    Returns:
        Fitted RandomForestClassifier.
    """
    print("\n[MODEL] Training Random Forest Classifier (supervised)...")
    print(f"        Estimators: {n_estimators} | Max depth: {max_depth or 'unlimited'}")

    start = time.time()
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,       # prevent over-splitting on tiny groups
        min_samples_leaf=2,        # each leaf needs at least 2 samples
        max_features="sqrt",       # classic RF: sqrt(n_features) per split
        class_weight="balanced",   # compensates for class imbalance (key for security!)
        bootstrap=True,
        n_jobs=-1,                 # use all CPU cores
        random_state=random_state,
        verbose=0,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    # Report training accuracy (use with caution — not a real measure)
    train_pred = model.predict(X_train)
    train_acc  = accuracy_score(y_train, train_pred)
    print(f"  [+] Training completed in {elapsed:.2f}s")
    print(f"  [+] Training accuracy: {train_acc*100:.2f}% (train-set, check test-set for true accuracy)")
    return model


# =============================================================================
# FUNCTION: train_logistic_regression
# Simple baseline model — demonstrates why complex models are needed.
# =============================================================================

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> LogisticRegression:
    """
    Train a Logistic Regression baseline model.

    PURPOSE:
      Logistic Regression is the "minimum viable classifier".
      It assumes a linear boundary between classes, which works
      poorly for complex attack patterns but gives us a baseline
      to justify using more powerful models.

    Args:
        X_train, y_train: Same as above.

    Returns:
        Fitted LogisticRegression model.
    """
    print("\n[MODEL] Training Logistic Regression (baseline)...")
    start = time.time()
    model = LogisticRegression(
        C=1.0,                     # regularization strength (higher = less regularization)
        solver="lbfgs",            # fast convergence for binary/multi-class
        max_iter=1000,             # enough iterations to converge
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  [+] Training completed in {elapsed:.2f}s")
    return model


# =============================================================================
# FUNCTION: evaluate_classifier
# Computes all standard cybersecurity ML metrics.
# =============================================================================

def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    class_names: list = None,
) -> dict:
    """
    Evaluate a classifier and print a full performance report.

    METRICS EXPLAINED:
      - Accuracy:  % of all predictions that are correct.
                   Can be misleading with class imbalance.
      - Precision: Of all "attack" predictions, what % were real attacks?
                   High precision → few false alarms (low FPR).
      - Recall:    Of all real attacks, what % did we catch?
                   High recall → few missed attacks.
      - F1-Score:  Harmonic mean of precision and recall. Best single metric
                   when you care about BOTH catching attacks AND avoiding false alarms.
      - AUC-ROC:   Area under the ROC curve. 1.0 = perfect, 0.5 = random.
                   Best for understanding detection vs. false positive trade-off.

    WHY ALL THESE MATTER IN SECURITY:
      A naive model that labels everything as "attack" gets 100% recall
      but 0% precision. SOC analysts care about both — too many false positives
      burn out analysts and make the system unusable.

    Args:
        model:       Any fitted sklearn classifier with predict() and predict_proba().
        X_test:      Scaled test feature matrix.
        y_test:      True labels for test set.
        model_name:  Label for printing.
        class_names: Optional list of class names for the report.

    Returns:
        dict with all computed metrics.
    """
    print(f"\n{'='*55}")
    print(f"  EVALUATION REPORT — {model_name}")
    print(f"{'='*55}")

    y_pred = model.predict(X_test)

    # ── Get probability scores (used for AUC-ROC) ────────────────────────────
    y_prob = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        # For binary: use probability of class 1 (attack)
        y_prob = proba[:, 1] if proba.shape[1] == 2 else proba

    # ── Core metrics ─────────────────────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"  Accuracy :  {acc*100:>6.2f}%")
    print(f"  Precision:  {prec*100:>6.2f}%")
    print(f"  Recall   :  {rec*100:>6.2f}%")
    print(f"  F1-Score :  {f1*100:>6.2f}%")

    # ── AUC-ROC (binary only) ─────────────────────────────────────────────────
    auc = None
    if y_prob is not None and len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_prob)
        print(f"  AUC-ROC  :  {auc:.4f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")

    # ── Detailed per-class report ─────────────────────────────────────────────
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0
    ))

    return {
        "model_name": model_name,
        "accuracy":   acc,
        "precision":  prec,
        "recall":     rec,
        "f1_score":   f1,
        "auc_roc":    auc,
        "confusion_matrix": cm,
        "y_pred":     y_pred,
        "y_prob":     y_prob,
    }


# =============================================================================
# FUNCTION: get_feature_importance
# Explains which network features matter most for threat detection.
# =============================================================================

def get_feature_importance(
    rf_model: RandomForestClassifier,
    feature_names: list,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extract the top-N most important features from Random Forest.

    WHY THIS MATTERS:
      Feature importance helps security teams understand:
        - Which network statistics are most predictive of attacks
        - Which CICFlowMeter features to prioritize in production
        - Evidence for why the model makes specific decisions (XAI)

    Common top features in network intrusion detection:
      - Flow Duration     : attacks often have unusual durations
      - Bwd Packet Length : response sizes differ in attacks
      - Flow Bytes/s      : DoS attacks have extreme rates
      - IAT Mean/Std      : inter-arrival time patterns differ
      - PSH/SYN flags     : protocol anomalies in scans

    Args:
        rf_model:      Fitted RandomForestClassifier.
        feature_names: List of feature names.
        top_n:         How many top features to return.

    Returns:
        DataFrame with Feature and Importance columns, sorted descending.
    """
    importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(top_n)

    print(f"\n[INFO] Top {top_n} most important features:")
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["Importance"] * 200)
        print(f"  {row['Feature']:<40} {row['Importance']:.4f}  {bar}")

    return importance_df.reset_index(drop=True)


# =============================================================================
# FUNCTION: cross_validate_model
# Verifies model isn't overfitting using k-fold cross validation.
# =============================================================================

def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "f1_weighted"
) -> dict:
    """
    Run k-fold cross-validation to check for overfitting.

    WHY CROSS-VALIDATION MATTERS:
      A model might look great on one train/test split by luck.
      CV splits the data k ways, trains and tests k times, then
      averages. Low variance across folds = model generalizes well.

    Args:
        model:    Any sklearn-compatible model.
        X, y:     Full feature matrix and labels (pre-scaling).
        cv:       Number of folds (5 is standard).
        scoring:  Metric to use. f1_weighted handles class imbalance.

    Returns:
        dict with 'scores', 'mean', 'std'.
    """
    print(f"\n[CV] Running {cv}-fold cross-validation ({scoring})...")
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    print(f"  Fold scores: {[round(s, 4) for s in scores]}")
    print(f"  Mean: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    return {"scores": scores, "mean": scores.mean(), "std": scores.std()}


# =============================================================================
# FUNCTION: save_model / load_model
# Persist and reload trained models to/from disk.
# =============================================================================

def save_model(model, filename: str) -> str:
    """
    Save a fitted model to disk using pickle.

    In production, you would use joblib instead of pickle for large models.
    MLflow or DVC would track model versions.

    Args:
        model:    Fitted sklearn model.
        filename: Filename (no path needed, saves to models/ dir).

    Returns:
        Full path where model was saved.
    """
    ensure_models_dir()
    filepath = os.path.join(MODELS_DIR, filename)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"  [SAVE] Model saved → {filepath}")
    return filepath


def load_model(filename: str):
    """
    Load a previously saved model from disk.

    Args:
        filename: Filename in the models/ directory.

    Returns:
        Loaded sklearn model ready for prediction.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}")
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print(f"  [LOAD] Model loaded ← {filepath}")
    return model


# =============================================================================
# FUNCTION: compare_models
# Builds a side-by-side comparison table of all trained models.
# =============================================================================

def compare_models(results: list) -> pd.DataFrame:
    """
    Build a comparison DataFrame from multiple evaluate_classifier() results.

    Args:
        results: List of dicts returned by evaluate_classifier().

    Returns:
        Formatted DataFrame showing metrics side by side.
    """
    rows = []
    for r in results:
        rows.append({
            "Model":     r["model_name"],
            "Accuracy":  f"{r['accuracy']*100:.2f}%",
            "Precision": f"{r['precision']*100:.2f}%",
            "Recall":    f"{r['recall']*100:.2f}%",
            "F1-Score":  f"{r['f1_score']*100:.2f}%",
            "AUC-ROC":   f"{r['auc_roc']:.4f}" if r["auc_roc"] else "N/A",
        })
    df = pd.DataFrame(rows)
    print("\n" + "="*60)
    print("  MODEL COMPARISON TABLE")
    print("="*60)
    print(df.to_string(index=False))
    return df


# =============================================================================
# FUNCTION: train_all_models
# Convenience wrapper — trains all three models and returns them.
# This is what main.py calls.
# =============================================================================

def train_all_models(
    X_train: np.ndarray,
    y_train_binary: np.ndarray,
    feature_names: list = None
) -> dict:
    """
    Train Isolation Forest, Random Forest, and Logistic Regression.

    Args:
        X_train:         Scaled training feature matrix.
        y_train_binary:  Binary labels (0=normal, 1=attack).
        feature_names:   Column names (for feature importance).

    Returns:
        dict with 'isolation_forest', 'random_forest', 'logistic_regression'.
    """
    print("\n" + "="*60)
    print("   MODEL TRAINING PIPELINE")
    print("="*60)

    # Train all three models
    iso_model = train_isolation_forest(X_train)
    rf_model  = train_random_forest(X_train, y_train_binary)
    lr_model  = train_logistic_regression(X_train, y_train_binary)

    # Save all models to disk
    print("\n[SAVE] Saving trained models...")
    save_model(iso_model, "isolation_forest.pkl")
    save_model(rf_model,  "random_forest.pkl")
    save_model(lr_model,  "logistic_regression.pkl")

    # Print feature importances
    if feature_names:
        get_feature_importance(rf_model, feature_names, top_n=15)

    print("\n[INFO] All models trained and saved successfully!")
    return {
        "isolation_forest":    iso_model,
        "random_forest":       rf_model,
        "logistic_regression": lr_model,
    }


# =============================================================================
# FUNCTION: evaluate_all_models
# Runs evaluation on all three models and compares them.
# =============================================================================

def evaluate_all_models(
    models: dict,
    X_test: np.ndarray,
    y_test_binary: np.ndarray,
    le_multi=None,
) -> dict:
    """
    Evaluate all trained models on the test set and compare.

    Args:
        models:         Dict of trained models (from train_all_models).
        X_test:         Scaled test features.
        y_test_binary:  Binary test labels.
        le_multi:       LabelEncoder for class names (optional).

    Returns:
        dict mapping model name → evaluation result dict.
    """
    print("\n" + "="*60)
    print("   MODEL EVALUATION PIPELINE")
    print("="*60)

    class_names = ["Normal", "Attack"]

    # ── Evaluate Random Forest ─────────────────────────────────────────────────
    rf_result = evaluate_classifier(
        models["random_forest"], X_test, y_test_binary,
        model_name="Random Forest", class_names=class_names
    )

    # ── Evaluate Logistic Regression ──────────────────────────────────────────
    lr_result = evaluate_classifier(
        models["logistic_regression"], X_test, y_test_binary,
        model_name="Logistic Regression (Baseline)", class_names=class_names
    )

    # ── Evaluate Isolation Forest (convert -1/+1 to 1/0) ─────────────────────
    iso_raw   = models["isolation_forest"].predict(X_test)
    iso_pred  = np.where(iso_raw == -1, 1, 0)   # Sklearn uses -1=anomaly, +1=normal
    acc_iso   = accuracy_score(y_test_binary, iso_pred)
    f1_iso    = f1_score(y_test_binary, iso_pred, average="weighted", zero_division=0)
    iso_result = {
        "model_name":       "Isolation Forest (Unsupervised)",
        "accuracy":         acc_iso,
        "precision":        precision_score(y_test_binary, iso_pred, average="weighted", zero_division=0),
        "recall":           recall_score(y_test_binary, iso_pred, average="weighted", zero_division=0),
        "f1_score":         f1_iso,
        "auc_roc":          None,
        "confusion_matrix": confusion_matrix(y_test_binary, iso_pred),
        "y_pred":           iso_pred,
        "y_prob":           None,
    }
    print(f"\n{'='*55}")
    print(f"  EVALUATION REPORT — Isolation Forest (Unsupervised)")
    print(f"{'='*55}")
    print(f"  Accuracy : {acc_iso*100:>6.2f}%")
    print(f"  F1-Score : {f1_iso*100:>6.2f}%")
    print(f"  Note: No predict_proba for IsolationForest (unsupervised)")

    # ── Side-by-side comparison ────────────────────────────────────────────────
    compare_models([rf_result, lr_result, iso_result])

    return {
        "random_forest":       rf_result,
        "logistic_regression": lr_result,
        "isolation_forest":    iso_result,
    }
