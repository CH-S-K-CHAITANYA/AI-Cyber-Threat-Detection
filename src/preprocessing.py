# =============================================================================
# FILE: src/preprocessing.py
# PURPOSE: Load, clean, encode, and scale the CICIDS-2017 network traffic dataset
#
# WHAT THIS FILE DOES (beginner explanation):
#   - Reads raw CSV files from the CICIDS-2017 dataset
#   - Removes missing/corrupted values that would break model training
#   - Converts text labels ("BENIGN", "DoS Hulk", etc.) into numbers
#   - Scales all feature values to a standard range so no single feature
#     dominates the model just because it has large numbers
#
# INDUSTRY RELEVANCE:
#   In real SOC (Security Operations Center) environments, raw network logs
#   arrive from tools like Zeek (Bro), Suricata, or Wireshark. They always
#   contain dirty data — missing packets, corrupt entries, duplicate flows.
#   Preprocessing is 60% of the real work in any ML security pipeline.
# =============================================================================

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# STEP 1: DATASET COLUMN REFERENCE
# These are the 78 features extracted by CICFlowMeter from network pcap files.
# Each row = one network flow (a conversation between two IP addresses).
# =============================================================================

# The CICIDS-2017 dataset label column name
LABEL_COLUMN = " Label"

# Attack categories present in the dataset and how we group them
ATTACK_MAPPING = {
    "BENIGN":                    "Normal",
    "DoS Hulk":                  "DoS",
    "DoS GoldenEye":             "DoS",
    "DoS slowloris":             "DoS",
    "DoS Slowhttptest":          "DoS",
    "DDoS":                      "DoS",
    "PortScan":                  "Port Scan",
    "FTP-Patator":               "Brute Force",
    "SSH-Patator":               "Brute Force",
    "Bot":                       "Botnet",
    "Web Attack – Brute Force":  "Web Attack",
    "Web Attack – XSS":          "Web Attack",
    "Web Attack – Sql Injection":"Web Attack",
    "Infiltration":              "Infiltration",
    "Heartbleed":                "Heartbleed",
}

# Binary mapping: 0 = Normal traffic, 1 = Attack traffic
BINARY_LABEL_MAP = {
    "Normal": 0,
    "DoS":          1,
    "Port Scan":    1,
    "Brute Force":  1,
    "Botnet":       1,
    "Web Attack":   1,
    "Infiltration": 1,
    "Heartbleed":   1,
}


# =============================================================================
# FUNCTION: load_dataset
# Loads one or multiple CICIDS-2017 CSV files into a single DataFrame.
# =============================================================================

def load_dataset(data_path: str, max_rows: int = None) -> pd.DataFrame:
    """
    Load CICIDS-2017 CSV file(s) into a pandas DataFrame.

    Args:
        data_path (str): Path to a single CSV file OR a folder of CSV files.
        max_rows (int):  Optional limit — useful when testing on a laptop
                         without loading all 2.8 million rows.

    Returns:
        pd.DataFrame: Combined raw dataframe.

    Example:
        df = load_dataset("data/raw/Friday-WorkingHours.pcap_ISCX.csv")
        df = load_dataset("data/raw/", max_rows=200000)
    """
    print("[LOAD] Reading dataset from:", data_path)

    # ── Case 1: path points to a single CSV file ──────────────────────────────
    if os.path.isfile(data_path) and data_path.endswith(".csv"):
        df = pd.read_csv(data_path, nrows=max_rows, low_memory=False)
        print(f"[LOAD] Loaded {len(df):,} rows × {df.shape[1]} columns")
        return df

    # ── Case 2: path points to a directory — load & concatenate all CSVs ──────
    elif os.path.isdir(data_path):
        csv_files = [
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".csv")
        ]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in: {data_path}")

        frames = []
        for fpath in csv_files:
            print(f"  [+] Loading {os.path.basename(fpath)} ...")
            chunk = pd.read_csv(fpath, nrows=max_rows, low_memory=False)
            frames.append(chunk)
        df = pd.concat(frames, ignore_index=True)
        print(f"[LOAD] Combined: {len(df):,} rows × {df.shape[1]} columns")
        return df

    else:
        raise FileNotFoundError(f"Path not found: {data_path}")


# =============================================================================
# FUNCTION: generate_demo_dataset
# Creates a synthetic dataset that MIRRORS real CICIDS-2017 statistics.
# USE THIS if you haven't downloaded the real dataset yet.
# Perfect for testing your code before downloading 7GB of data.
# =============================================================================

def generate_demo_dataset(n_samples: int = 50000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic network traffic dataset with realistic distributions.
    Mirrors the statistical profile of CICIDS-2017.

    Args:
        n_samples (int): Total number of flow records to generate.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: Synthetic dataset ready for the full pipeline.
    """
    print(f"[DEMO] Generating synthetic dataset with {n_samples:,} samples...")
    rng = np.random.default_rng(random_state)

    # ── Decide how many of each class we want (realistic imbalance) ───────────
    # Real CICIDS-2017 is ~80% benign, ~20% attacks
    n_benign    = int(n_samples * 0.78)
    n_dos       = int(n_samples * 0.09)
    n_portscan  = int(n_samples * 0.06)
    n_bruteforce= int(n_samples * 0.04)
    n_webattack = int(n_samples * 0.02)
    n_botnet    = n_samples - n_benign - n_dos - n_portscan - n_bruteforce - n_webattack

    # ── Helper: add noise to a base value ─────────────────────────────────────
    def noisy(base, scale, size):
        return np.abs(rng.normal(base, scale, size))

    # ── Build feature blocks per attack type ──────────────────────────────────
    # Features: Flow Duration, Total Fwd Packets, Bwd Packets, Fwd Bytes,
    #           Bwd Bytes, Flow Bytes/s, Flow Packets/s, IAT Mean, IAT Std,
    #           Fwd IAT Mean, Active Mean, Idle Mean, PSH Flags, SYN Flags,
    #           RST Flags, FIN Flags, Subflow Fwd Bytes, init_win_bytes_fwd

    def make_block(n, label, dur_base, fwd_pkts_base, bwd_pkts_base,
                   bps_base, pps_base, iat_mean_base, psh, syn, rst):
        return pd.DataFrame({
            " Flow Duration":        noisy(dur_base, dur_base*0.3, n),
            " Total Fwd Packets":    noisy(fwd_pkts_base, fwd_pkts_base*0.4, n).astype(int),
            " Total Backward Packets": noisy(bwd_pkts_base, bwd_pkts_base*0.4, n).astype(int),
            "Total Length of Fwd Packets": noisy(fwd_pkts_base*500, 2000, n),
            " Total Length of Bwd Packets": noisy(bwd_pkts_base*400, 1500, n),
            " Flow Bytes/s":         noisy(bps_base, bps_base*0.5, n),
            " Flow Packets/s":       noisy(pps_base, pps_base*0.5, n),
            " Flow IAT Mean":        noisy(iat_mean_base, iat_mean_base*0.4, n),
            " Flow IAT Std":         noisy(iat_mean_base*0.5, iat_mean_base*0.3, n),
            " Fwd IAT Mean":         noisy(iat_mean_base*0.8, iat_mean_base*0.3, n),
            " Bwd IAT Mean":         noisy(iat_mean_base*1.2, iat_mean_base*0.4, n),
            " Fwd PSH Flags":        rng.binomial(1, psh, n),
            " SYN Flag Count":       rng.binomial(3, syn, n),
            " RST Flag Count":       rng.binomial(2, rst, n),
            " FIN Flag Count":       rng.binomial(2, 0.3, n),
            " ACK Flag Count":       rng.binomial(5, 0.6, n),
            " Average Packet Size":  noisy(fwd_pkts_base*40, 100, n),
            " Avg Fwd Segment Size": noisy(fwd_pkts_base*45, 80, n),
            " Active Mean":          noisy(dur_base*0.6, dur_base*0.2, n),
            " Idle Mean":            noisy(dur_base*0.2, dur_base*0.1, n),
            " Subflow Fwd Bytes":    noisy(fwd_pkts_base*500, 1000, n),
            " init_win_bytes_forward": noisy(8192, 2000, n).astype(int),
            " init_win_bytes_backward": noisy(4096, 1000, n).astype(int),
            " Destination Port":     rng.integers(1, 65535, n),
            " Protocol":             rng.choice([6, 17, 0], n, p=[0.6, 0.3, 0.1]),
            " Label":                [label] * n,
        })

    # Each attack type has a distinct statistical fingerprint:
    #  - DoS: extremely high packet rate, low IAT, many flows per second
    #  - Port Scan: many RST/SYN flags, short durations, many destinations
    #  - Brute Force: repetitive pattern, medium IAT, focused port
    #  - Web Attack: medium-large packets, focused on port 80/443/8080
    #  - Botnet: low and slow, irregular IAT, PSH flags

    frames = [
        make_block(n_benign,    "BENIGN",                 50000, 10,  8,  5000,  20,  3000, 0.3, 0.1, 0.05),
        make_block(n_dos,       "DoS GoldenEye",           1000,100,  0, 900000, 800,    50, 0.1, 0.9, 0.3),
        make_block(n_portscan,  "PortScan",                 500,  2,  0,  2000,  30,   100, 0.0, 0.9, 0.8),
        make_block(n_bruteforce,"FTP-Patator",             3000, 12,  8,  8000,  40,   800, 0.4, 0.2, 0.1),
        make_block(n_webattack, "Web Attack \x96 Brute Force", 8000, 15, 10, 15000,  25,  2000, 0.5, 0.1, 0.05),
        make_block(n_botnet,    "Bot",                    40000,  5,  3,  1000,   8,  5000, 0.6, 0.1, 0.05),
    ]

    df = pd.concat(frames, ignore_index=True)

    # Shuffle so classes are mixed
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"[DEMO] Generated {len(df):,} rows | Class distribution:")
    for lbl, cnt in df[" Label"].value_counts().items():
        print(f"       {lbl:<35} → {cnt:>6,} rows ({cnt/len(df)*100:.1f}%)")

    return df


# =============================================================================
# FUNCTION: clean_data
# Handles all data quality issues found in CICIDS-2017.
# =============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw CICIDS-2017 data.

    Issues addressed:
      1. Column name whitespace (CICIDS-2017 has leading spaces in headers)
      2. Infinite values (division-by-zero in CICFlowMeter calculations)
      3. NaN values (dropped connections, incomplete flows)
      4. Duplicate rows (same flow captured twice)
      5. Negative values in columns that should always be positive

    Args:
        df (pd.DataFrame): Raw loaded dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    print("[CLEAN] Starting data cleaning...")
    original_shape = df.shape

    # ── 1. Strip leading/trailing whitespace from column names ────────────────
    # CICIDS-2017 has headers like " Flow Duration" (with a leading space)
    df.columns = df.columns.str.strip()
    print(f"  [+] Stripped whitespace from {df.shape[1]} column names")

    # ── 2. Replace infinite values with NaN (then drop them) ─────────────────
    # CICFlowMeter computes Flow Bytes/s = bytes / duration.
    # If duration = 0 microseconds, this becomes infinity.
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"  [+] Replaced {inf_count:,} infinite values with NaN")

    # ── 3. Drop rows with any NaN ─────────────────────────────────────────────
    nan_count = df.isnull().sum().sum()
    df.dropna(inplace=True)
    print(f"  [+] Dropped {nan_count:,} NaN cells ({original_shape[0]-len(df):,} rows removed)")

    # ── 4. Remove exact duplicate rows ───────────────────────────────────────
    dup_count = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"  [+] Dropped {dup_count:,} duplicate rows")

    # ── 5. Remove rows with negative values in inherently positive columns ────
    non_negative_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if any(keyword in c.lower() for keyword in
               ["bytes", "packets", "duration", "length", "size"])
    ]
    mask = (df[non_negative_cols] < 0).any(axis=1)
    neg_count = mask.sum()
    df = df[~mask]
    print(f"  [+] Removed {neg_count:,} rows with negative values in {len(non_negative_cols)} columns")

    # ── 6. Fix the Label column name (strip whitespace) ───────────────────────
    if "Label" not in df.columns and " Label" in df.columns:
        df.rename(columns={" Label": "Label"}, inplace=True)

    print(f"[CLEAN] Done. Shape: {original_shape} → {df.shape}")
    return df.reset_index(drop=True)


# =============================================================================
# FUNCTION: encode_labels
# Converts string attack labels into numeric form for ML models.
# =============================================================================

def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Encode the 'Label' column in three ways:
      1. Grouped category label   (e.g., "DoS", "Brute Force")
      2. Binary label             (0 = Normal, 1 = Attack)
      3. Multi-class integer      (e.g., 0=Normal, 1=DoS, 2=PortScan, ...)

    Args:
        df (pd.DataFrame): Cleaned dataframe with 'Label' column.

    Returns:
        tuple: (df_with_encodings, binary_encoder, multiclass_encoder)
    """
    print("[ENCODE] Encoding attack labels...")

    # ── Map raw labels to grouped categories ─────────────────────────────────
    def map_label(raw_label: str) -> str:
        """Map a raw CICIDS label to our grouped category."""
        raw_clean = raw_label.strip()
        for key, value in ATTACK_MAPPING.items():
            if key.lower() in raw_clean.lower():
                return value
        return "Unknown"

    df["attack_category"] = df["Label"].apply(map_label)

    # ── Binary label: 0 (normal) or 1 (any attack) ────────────────────────────
    df["label_binary"] = df["attack_category"].map(BINARY_LABEL_MAP).fillna(1).astype(int)

    # ── Multi-class integer encoding ──────────────────────────────────────────
    le_multi = LabelEncoder()
    df["label_multiclass"] = le_multi.fit_transform(df["attack_category"])

    # ── Binary LabelEncoder (for inverse_transform later) ────────────────────
    le_binary = LabelEncoder()
    le_binary.fit(["Normal", "Attack"])

    # Print distribution
    print(f"[ENCODE] Binary distribution:")
    print(f"         Normal (0): {(df['label_binary']==0).sum():,}")
    print(f"         Attack (1): {(df['label_binary']==1).sum():,}")
    print(f"[ENCODE] Attack categories found:")
    for cat, cnt in df["attack_category"].value_counts().items():
        print(f"         {cat:<20} → {cnt:>6,}")

    return df, le_binary, le_multi


# =============================================================================
# FUNCTION: select_features
# Picks the most informative features for training.
# Removes low-variance and highly correlated redundant features.
# =============================================================================

def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Select the most informative numeric features.

    Removes:
      - Non-numeric columns (except Label)
      - Zero-variance columns (same value in every row — useless for ML)
      - Highly correlated pairs (keep one, drop the other)

    Args:
        df (pd.DataFrame): Cleaned + encoded dataframe.

    Returns:
        tuple: (df_with_selected_features, list_of_selected_feature_names)
    """
    print("[FEATURE] Selecting features...")

    # ── Get only numeric columns, excluding label columns ────────────────────
    label_cols = ["Label", "attack_category", "label_binary", "label_multiclass"]
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in label_cols
    ]
    print(f"  [+] Starting with {len(numeric_cols)} numeric features")

    # ── Drop zero-variance columns ────────────────────────────────────────────
    variances = df[numeric_cols].var()
    zero_var_cols = variances[variances == 0].index.tolist()
    numeric_cols = [c for c in numeric_cols if c not in zero_var_cols]
    print(f"  [+] Dropped {len(zero_var_cols)} zero-variance columns")

    # ── Drop highly correlated columns (threshold = 0.97) ────────────────────
    corr_matrix = df[numeric_cols].corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_cols = [
        col for col in upper_tri.columns
        if any(upper_tri[col] > 0.97)
    ]
    numeric_cols = [c for c in numeric_cols if c not in high_corr_cols]
    print(f"  [+] Dropped {len(high_corr_cols)} highly correlated columns")
    print(f"  [+] Final feature count: {len(numeric_cols)}")

    return df[numeric_cols + label_cols], numeric_cols


# =============================================================================
# FUNCTION: scale_features
# Standardizes features to zero mean, unit variance.
# Critical for distance-based models like Isolation Forest.
# =============================================================================

def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Apply StandardScaler: (value - mean) / std_dev

    IMPORTANT: Scaler is FIT only on training data to prevent data leakage.
    The same fitted scaler is then applied to test data.

    Args:
        X_train: Training feature matrix
        X_test:  Test feature matrix

    Returns:
        tuple: (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    print("[SCALE] Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
    X_test_scaled  = scaler.transform(X_test)         # transform-only on test
    print(f"  [+] Train set scaled: {X_train_scaled.shape}")
    print(f"  [+] Test set scaled:  {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, scaler


# =============================================================================
# FUNCTION: full_preprocessing_pipeline
# Runs all steps in sequence. This is what main.py calls.
# =============================================================================

def full_preprocessing_pipeline(
    data_path: str = None,
    max_rows: int = None,
    test_size: float = 0.2,
    use_demo: bool = False,
    demo_samples: int = 50000
) -> dict:
    """
    Complete preprocessing pipeline from raw data to train/test splits.

    Args:
        data_path (str):    Path to CSV file or folder.
        max_rows (int):     Limit rows loaded (None = all).
        test_size (float):  Fraction for test split (default 20%).
        use_demo (bool):    If True, generate synthetic data instead.
        demo_samples (int): How many samples to generate for demo mode.

    Returns:
        dict with keys:
            'X_train', 'X_test', 'y_train_binary', 'y_test_binary',
            'y_train_multi', 'y_test_multi', 'scaler', 'feature_names',
            'le_binary', 'le_multi', 'df_processed'
    """
    print("\n" + "="*60)
    print("   PREPROCESSING PIPELINE")
    print("="*60)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    if use_demo or data_path is None:
        print("[INFO] Using synthetic demo dataset (real dataset not provided)")
        df = generate_demo_dataset(n_samples=demo_samples)
    else:
        df = load_dataset(data_path, max_rows=max_rows)

    # ── Step 2: Clean data ────────────────────────────────────────────────────
    df = clean_data(df)

    # ── Step 3: Encode labels ─────────────────────────────────────────────────
    df, le_binary, le_multi = encode_labels(df)

    # ── Step 4: Select features ───────────────────────────────────────────────
    df, feature_names = select_features(df)

    # ── Step 5: Prepare feature matrix and label vectors ─────────────────────
    X = df[feature_names].values
    y_binary = df["label_binary"].values
    y_multi  = df["label_multiclass"].values

    print(f"\n[SPLIT] Creating train/test split ({int((1-test_size)*100)}/{int(test_size*100)})...")
    X_train, X_test, y_train_b, y_test_b, y_train_m, y_test_m = train_test_split(
        X, y_binary, y_multi,
        test_size=test_size,
        random_state=42,
        stratify=y_binary        # keep class balance in both splits
    )

    # ── Step 6: Scale features ────────────────────────────────────────────────
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    print("\n[PIPELINE] Preprocessing complete!")
    print(f"  Train: {X_train_sc.shape[0]:,} samples | Test: {X_test_sc.shape[0]:,} samples")
    print(f"  Features: {len(feature_names)}")
    print(f"  Attack rate (train): {y_train_b.mean()*100:.1f}%")

    return {
        "X_train":         X_train_sc,
        "X_test":          X_test_sc,
        "y_train_binary":  y_train_b,
        "y_test_binary":   y_test_b,
        "y_train_multi":   y_train_m,
        "y_test_multi":    y_test_m,
        "scaler":          scaler,
        "feature_names":   feature_names,
        "le_binary":       le_binary,
        "le_multi":        le_multi,
        "df_processed":    df,
    }
