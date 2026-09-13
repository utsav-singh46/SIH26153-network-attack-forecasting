import os
import numpy as np
import pandas as pd

# Paths to cleaned input dataset and aggregated network states output
CLEANED_DATA_PATH = os.path.join("data", "processed", "cleaned_traffic.csv")
OUTPUT_DATA_PATH = os.path.join("data", "processed", "network_states_30s.csv")

# Fixed temporal window size (30 seconds)
WINDOW_SIZE = "30s"


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
    Groups individual network flow records into 30-second temporal windows
    and computes aggregated network-state features and window-level target labels.
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
    else:
        print("Note: Destination Port column not found. Skipping unique_destination_ports.")

    if col_protocol:
        aggregations["unique_protocols"] = (col_protocol, "nunique")
    else:
        print("Note: Protocol column not found. Skipping unique_protocols.")

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

    # Perform window aggregation
    window_df = resampled.agg(**aggregations).reset_index()
    window_df.rename(columns={"Timestamp": "window_start"}, inplace=True)

    # Filter out empty time periods (windows with zero captured flows)
    window_df = window_df[window_df["flow_count"] > 0].copy()

    # Calculate proportion of infiltration flows within each 30-second window
    window_df["infiltration_ratio"] = window_df["infiltration_flows"] / window_df["flow_count"]

    # Assign window label: "Infiltration" if any flow in the window was an attack, else "Benign"
    window_df["Label"] = np.where(window_df["infiltration_flows"] > 0, "Infiltration", "Benign")

    # Fill NaNs created by zero-duration divisions with 0
    numeric_cols = window_df.select_dtypes(include=[np.number]).columns
    window_df[numeric_cols] = window_df[numeric_cols].fillna(0)

    return window_df


def generate_and_save_features():
    """Main execution entry point for feature extraction and validation."""
    print("Loading preprocessed flow data...")
    df_raw = load_cleaned_dataset(CLEANED_DATA_PATH)
    original_flow_rows = len(df_raw)
    original_infiltration_rows = int(df_raw["is_infiltration"].sum())

    print("Grouping network flows into 30-second network-state windows...")
    df_windowed = aggregate_network_states(df_raw)

    # Critical Validation Checks
    reconciled_infiltration_sum = int(df_windowed["infiltration_flows"].sum())
    windows_with_infiltration = (df_windowed["infiltration_flows"] > 0).sum()

    print("\n--- Critical Validation Check ---")
    print(f"Total Infiltration Rows in Input : {original_infiltration_rows}")
    print(f"Sum of Infiltration Flows        : {reconciled_infiltration_sum}")
    print(f"Infiltration Flow Reconciliation : {original_infiltration_rows == reconciled_infiltration_sum}")

    print("\nSample Infiltration Windows (5 Examples):")
    sample_cols = ["window_start", "flow_count", "infiltration_flows", "infiltration_ratio", "Label"]
    sample_windows = df_windowed[df_windowed["infiltration_flows"] > 0][sample_cols].head(5)
    print(sample_windows.to_string(index=False))

    # Drop intermediate counting helper column before final export
    df_export = df_windowed.drop(columns=["infiltration_flows"])

    # Save aggregated 30-second network states
    os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok=True)
    df_export.to_csv(OUTPUT_DATA_PATH, index=False)

    # Print Final Output Summary
    total_windows = len(df_export)
    benign_windows = (df_export["Label"] == "Benign").sum()
    infiltration_windows = (df_export["Label"] == "Infiltration").sum()
    min_ratio = df_export["infiltration_ratio"].min()
    max_ratio = df_export["infiltration_ratio"].max()

    print("\n--- Final Network State Extraction Summary ---")
    print(f"Original Flow Rows          : {original_flow_rows}")
    print(f"30-Second Windows Generated : {total_windows}")
    print(f"Start Window                : {df_export['window_start'].min()}")
    print(f"End Window                  : {df_export['window_start'].max()}")
    print(f"Benign Windows              : {benign_windows}")
    print(f"Infiltration Windows        : {infiltration_windows}")
    print(f"Min Infiltration Ratio      : {min_ratio:.4f}")
    print(f"Max Infiltration Ratio      : {max_ratio:.4f}")
    print(f"Total Infiltration Flows    : {reconciled_infiltration_sum}")
    print(f"Total Feature Columns       : {df_export.shape[1]}")
    print("\nFinal Column Names:")
    for col in df_export.columns:
        print(f" - {col}")
    print(f"\nSaved 30-second network states to: {OUTPUT_DATA_PATH}\n")


if __name__ == "__main__":
    generate_and_save_features()