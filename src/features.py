import os
import numpy as np
import pandas as pd

# Path definitions
CLEANED_DATA_PATH = os.path.join("data", "processed", "cleaned_traffic.csv")
NETWORK_STATES_PATH = os.path.join("data", "processed", "network_states_30s.csv")
X_SEQUENCES_PATH = os.path.join("data", "processed", "X_sequences.npy")
Y_SEQUENCES_PATH = os.path.join("data", "processed", "y_sequences.npy")
METADATA_PATH = os.path.join("data", "processed", "sequence_metadata.csv")

# Configuration constants
WINDOW_SIZE = "30s"
SEQUENCE_LENGTH = 5  # Number of past 30-second states used to predict next state label


def load_cleaned_dataset(file_path: str) -> pd.DataFrame:
    """Loads preprocessed traffic flows, parses timestamps, and sets up infiltration flags."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {file_path}")

    # Load preprocessed network traffic CSV
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()

    # Normalize Label strings to ensure consistent equality checks
    df["Label"] = df["Label"].astype(str).str.strip()

    # Create explicit integer flag for infiltration flows (1 if Infiltration, 0 otherwise)
    df["is_infiltration"] = (df["Label"] == "Infiltration").astype(int)

    # Parse timestamps to datetime and sort chronologically
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(by="Timestamp").reset_index(drop=True)

    return df


def match_column(df_columns: pd.Index, candidates: list) -> str:
    """Helper function to find matching CICFlowMeter feature column names dynamically."""
    for candidate in candidates:
        if candidate in df_columns:
            return candidate
    return None


def aggregate_network_states(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups individual network flow records into continuous 30-second temporal windows.
    Preserves all empty/zero-flow intervals to maintain chronological timeline integrity.
    """
    cols = df.columns

    # Dynamic column mapping to account for potential header variations in CICFlowMeter datasets
    col_fwd_pkts = match_column(cols, ["Tot Fwd Pkts", "Total Fwd Packets"])
    col_bwd_pkts = match_column(cols, ["Tot Bwd Pkts", "Total Backward Packets"])
    col_fwd_bytes = match_column(cols, ["TotLen Fwd Pkts", "Total Length of Fwd Packets"])
    col_bwd_bytes = match_column(cols, ["TotLen Bwd Pkts", "Total Length of Bwd Packets"])
    col_duration = match_column(cols, ["Flow Duration"])
    col_bytes_s = match_column(cols, ["Flow Byts/s", "Flow Bytes/s"])
    col_pkts_s = match_column(cols, ["Flow Pkts/s", "Flow Packets/s"])
    col_fwd_len_mean = match_column(cols, ["Fwd Pkt Len Mean", "Fwd Packet Length Mean"])
    col_bwd_len_mean = match_column(cols, ["Bwd Pkt Len Mean", "Bwd Packet Length Mean"])
    col_dst_port = match_column(cols, ["Dst Port", "Destination Port"])
    col_protocol = match_column(cols, ["Protocol"])

    # Prepare numeric helper columns directly on the DataFrame for sum aggregations
    df["_fwd_pkts"] = pd.to_numeric(df[col_fwd_pkts], errors="coerce").fillna(0) if col_fwd_pkts else 0
    df["_bwd_pkts"] = pd.to_numeric(df[col_bwd_pkts], errors="coerce").fillna(0) if col_bwd_pkts else 0
    df["_total_pkts"] = df["_fwd_pkts"] + df["_bwd_pkts"]

    df["_fwd_bytes"] = pd.to_numeric(df[col_fwd_bytes], errors="coerce").fillna(0) if col_fwd_bytes else 0
    df["_bwd_bytes"] = pd.to_numeric(df[col_bwd_bytes], errors="coerce").fillna(0) if col_bwd_bytes else 0
    df["_total_bytes"] = df["_fwd_bytes"] + df["_bwd_bytes"]

    # Set Timestamp index for temporal pandas resample operations
    df_indexed = df.set_index("Timestamp")
    resampled = df_indexed.resample(WINDOW_SIZE)

    # Core aggregation mappings
    aggregations = {
        "flow_count": ("Label", "count"),
        "infiltration_flows": ("is_infiltration", "sum"),
        "total_forward_packets": ("_fwd_pkts", "sum"),
        "total_backward_packets": ("_bwd_pkts", "sum"),
        "total_packets": ("_total_pkts", "sum"),
        "total_forward_bytes": ("_fwd_bytes", "sum"),
        "total_backward_bytes": ("_bwd_bytes", "sum"),
        "total_bytes": ("_total_bytes", "sum"),
    }

    # Unique entity counts
    if col_dst_port:
        aggregations["unique_destination_ports"] = (col_dst_port, "nunique")
    if col_protocol:
        aggregations["unique_protocols"] = (col_protocol, "nunique")

    # Flow statistical averages and medians
    if col_duration:
        aggregations["average_flow_duration"] = (col_duration, "mean")
        aggregations["median_flow_duration"] = (col_duration, "median")
    if col_bytes_s:
        aggregations["average_flow_bytes_per_second"] = (col_bytes_s, "mean")
    if col_pkts_s:
        aggregations["average_flow_packets_per_second"] = (col_pkts_s, "mean")
    if col_fwd_len_mean:
        aggregations["average_forward_packet_length"] = (col_fwd_len_mean, "mean")
    if col_bwd_len_mean:
        aggregations["average_backward_packet_length"] = (col_bwd_len_mean, "mean")

    # Perform window aggregation across the complete continuous timeframe
    window_df = resampled.agg(**aggregations).reset_index()
    window_df.rename(columns={"Timestamp": "window_start"}, inplace=True)

    # DO NOT DROP ZERO-FLOW WINDOWS: Keep all 30-second windows intact
    # Calculate infiltration_ratio safely for empty/zero-flow windows
    window_df["infiltration_ratio"] = np.where(
        window_df["flow_count"] > 0,
        window_df["infiltration_flows"] / window_df["flow_count"],
        0.0
    )

    # Assign window label: "Infiltration" if any attack flow occurred, else "Benign"
    window_df["Label"] = np.where(window_df["infiltration_flows"] > 0, "Infiltration", "Benign")

    # Fill NaNs across all feature metrics (zero-flow windows get 0 values)
    numeric_cols = window_df.select_dtypes(include=[np.number]).columns
    window_df[numeric_cols] = window_df[numeric_cols].fillna(0)

    return window_df


def create_temporal_sequences(states_df: pd.DataFrame, seq_len: int = 5):
    """
    Converts strictly continuous 30-second network state snapshots into sliding window sequence arrays (X, y)
    for next-state network attack forecasting.

    Input features: Only numeric network state metrics (excluding window_start, Label, infiltration_ratio).
    Target y      : Next window's network attack state label (Benign -> 0, Infiltration -> 1).
    """
    # 1. Sort strictly chronologically by window_start
    df = states_df.sort_values(by="window_start").reset_index(drop=True)

    # 2. Exclude non-feature or data-leakage columns from model inputs
    excluded_cols = ["window_start", "Label", "infiltration_ratio"]
    feature_cols = [c for c in df.columns if c not in excluded_cols]

    # Extract feature matrix (NumPy float32 array)
    X_matrix = df[feature_cols].values.astype(np.float32)

    # Map target Labels to integers: Benign -> 0, Infiltration -> 1
    label_mapping = {"Benign": 0, "Infiltration": 1}
    y_vector = df["Label"].map(label_mapping).values

    num_samples = len(df)
    X_seq_list = []
    y_seq_list = []
    metadata_records = []

    # 3. Construct sliding window sequences: [t-4, t-3, t-2, t-1, t] -> target [t+1]
    for i in range(num_samples - seq_len):
        # Input sequence matrix of shape (seq_len, num_features)
        X_seq = X_matrix[i : i + seq_len]
        
        # Target label corresponding to state at index i + seq_len (t + 1)
        y_target = y_vector[i + seq_len]

        X_seq_list.append(X_seq)
        y_seq_list.append(y_target)

        # Metadata tracking for validation and temporal alignment
        metadata_records.append({
            "sequence_index": i,
            "input_start_window": df.loc[i, "window_start"],
            "input_end_window": df.loc[i + seq_len - 1, "window_start"],
            "target_window": df.loc[i + seq_len, "window_start"],
            "target_label": df.loc[i + seq_len, "Label"],
            "target_label_encoded": y_target,
        })

    # Convert to NumPy arrays while strictly maintaining chronological order
    X_arr = np.array(X_seq_list, dtype=np.float32)
    y_arr = np.array(y_seq_list, dtype=np.int64)
    metadata_df = pd.DataFrame(metadata_records)

    return X_arr, y_arr, metadata_df, feature_cols


def run_feature_and_sequence_generation():
    """Main workflow: Generates continuous 30s network states followed by temporal forecasting sequences."""
    print("Step 1: Loading preprocessed traffic flows...")
    df_raw = load_cleaned_dataset(CLEANED_DATA_PATH)
    original_flow_rows = len(df_raw)
    original_infiltration_rows = int(df_raw["is_infiltration"].sum())

    print("Step 2: Aggregating network flows into continuous 30-second network states...")
    df_windowed = aggregate_network_states(df_raw)

    # Reconcile raw flow infiltration sum
    reconciled_infiltration = int(df_windowed["infiltration_flows"].sum())
    assert original_infiltration_rows == reconciled_infiltration, "Infiltration flow sum mismatch!"

    # Temporal continuity validation: Check difference between consecutive window timestamps
    window_diffs = df_windowed["window_start"].diff().dropna()
    min_gap = window_diffs.min()
    max_gap = window_diffs.max()
    is_continuous_30s = (min_gap == pd.Timedelta(seconds=30)) and (max_gap == pd.Timedelta(seconds=30))

    zero_flow_windows = (df_windowed["flow_count"] == 0).sum()
    non_empty_windows = (df_windowed["flow_count"] > 0).sum()
    total_windows = len(df_windowed)

    print("\n--- Temporal Continuity Validation ---")
    print(f"Total 30-Second Windows    : {total_windows}")
    print(f"Non-Empty Windows          : {non_empty_windows}")
    print(f"Zero-Flow Windows          : {zero_flow_windows}")
    print(f"Minimum Window Gap         : {min_gap}")
    print(f"Maximum Window Gap         : {max_gap}")
    print(f"Strict 30-Second Continuity: {is_continuous_30s}")

    assert is_continuous_30s, "Validation Failed: Timeline contains non-30-second gaps!"

    # Drop temporary counting helper column before export
    df_export = df_windowed.drop(columns=["infiltration_flows"])

    # Save 30-second network states CSV
    os.makedirs(os.path.dirname(NETWORK_STATES_PATH), exist_ok=True)
    df_export.to_csv(NETWORK_STATES_PATH, index=False)
    print(f"\nSaved complete network states to: {NETWORK_STATES_PATH}")

    # Step 3: Sequence creation for temporal forecasting
    print("\nStep 3: Creating sliding window temporal sequences (sequence_length = 5)...")
    X, y, metadata_df, feature_names = create_temporal_sequences(df_export, seq_len=SEQUENCE_LENGTH)

    # Save numpy arrays and metadata
    np.save(X_SEQUENCES_PATH, X)
    np.save(Y_SEQUENCES_PATH, y)
    metadata_df.to_csv(METADATA_PATH, index=False)

    print(f"Saved sequence inputs to: {X_SEQUENCES_PATH}")
    print(f"Saved sequence targets to: {Y_SEQUENCES_PATH}")
    print(f"Saved sequence metadata to: {METADATA_PATH}")

    # Target distribution summary
    unique_labels, counts = np.unique(y, return_counts=True)
    target_counts_dict = dict(zip(unique_labels, counts))

    print("\n--- Temporal Sequence Generation Summary ---")
    print(f"Number of Network States : {len(df_export)}")
    print(f"Sequence Length          : {SEQUENCE_LENGTH}")
    print(f"Number of Input Features : {X.shape[2]}")
    print(f"Number of Sequences      : {len(X)}")
    print(f"X Shape                  : {X.shape}")
    print(f"y Shape                  : {y.shape}")
    print(f"Target Label Counts (y)  : Benign (0): {target_counts_dict.get(0, 0)}, Infiltration (1): {target_counts_dict.get(1, 0)}")

    print("\nModel Input Feature Columns:")
    for fn in feature_names:
        print(f" - {fn}")
    print("\nProcessing complete.\n")


if __name__ == "__main__":
    run_feature_and_sequence_generation()